// Supabase Edge Function: question-ai
// Runs on Supabase's servers. The OpenAI key is read from the OPENAI_API_KEY
// secret and never leaves this function.
//
// Actions (POST JSON, signed-in users only):
//   { action: "extract", upload_id }   -> reads one screenshot, returns questions
//   { action: "dedupe",  drafts: [...] } -> finds similar questions already saved
//
// Secrets (Supabase dashboard -> Edge Functions -> Secrets):
//   OPENAI_API_KEY      required
//   OPENAI_MODEL        optional, default "gpt-6.2-sol"  (reads screenshots)
//   OPENAI_DUP_MODEL    optional, default "gpt-6-luna"   (duplicate checks)
//   AI_DAILY_LIMIT      optional, default 40 screenshots per user per day

import { createClient } from "npm:@supabase/supabase-js@2";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
const BUCKET = "question-images";
const env = (k: string, d = "") => Deno.env.get(k) ?? d;

function reply(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS, "Content-Type": "application/json" },
  });
}

// The public key Supabase gives every function. Newer projects expose
// SUPABASE_PUBLISHABLE_KEYS (JSON), older ones SUPABASE_ANON_KEY.
function publicKey() {
  const many = env("SUPABASE_PUBLISHABLE_KEYS");
  if (many) {
    try {
      const o = JSON.parse(many);
      const v = o.default ?? Object.values(o)[0];
      if (typeof v === "string" && v) return v;
    } catch { /* fall through */ }
  }
  return env("SUPABASE_PUBLISHABLE_KEY") || env("SUPABASE_ANON_KEY");
}

class UserError extends Error {
  status: number;
  constructor(msg: string, status = 400) { super(msg); this.status = status; }
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });
  if (req.method !== "POST") return reply({ error: "Use POST" }, 405);

  try {
    const auth = req.headers.get("Authorization") ?? "";
    const token = auth.replace(/^Bearer\s+/i, "");
    if (!token) throw new UserError("Sign in first.", 401);

    const key = publicKey();
    if (!key) throw new UserError("Supabase did not provide a public API key to this function.", 500);
    const sb = createClient(env("SUPABASE_URL"), key, {
      global: { headers: { Authorization: `Bearer ${token}` } },
      auth: { persistSession: false, autoRefreshToken: false },
    });
    const { data: u, error: ue } = await sb.auth.getUser(token);
    if (ue || !u?.user) throw new UserError("Your session has expired. Sign in again.", 401);

    if (!env("OPENAI_API_KEY")) {
      throw new UserError("The OPENAI_API_KEY secret is not set in Supabase.", 500);
    }

    const body = await req.json().catch(() => ({}));
    if (body.action === "extract") return reply(await extract(sb, body));
    if (body.action === "dedupe") return reply(await dedupe(sb, body));
    throw new UserError("Unknown action.");
  } catch (e) {
    const status = e instanceof UserError ? e.status : 500;
    console.error(e);
    return reply({ error: e instanceof Error ? e.message : String(e) }, status);
  }
});

// ------------------------------------------------------------ OpenAI call

async function openai(model: string, content: unknown[], schemaName: string, schema: unknown, instructions: string) {
  const payload: Record<string, unknown> = {
    model,
    instructions,
    input: [{ role: "user", content }],
    text: { format: { type: "json_schema", name: schemaName, schema, strict: true } },
  };
  const effort = env("OPENAI_REASONING", "low");
  if (effort !== "none") payload.reasoning = { effort };

  let res = await callOpenAI(payload);
  // Some models do not take a reasoning setting: retry once without it.
  if (res.status === 400 && payload.reasoning) {
    const txt = await res.clone().text();
    if (/reasoning/i.test(txt)) {
      delete payload.reasoning;
      res = await callOpenAI(payload);
    }
  }
  if (!res.ok) {
    const txt = await res.text();
    let msg = txt;
    try { msg = JSON.parse(txt)?.error?.message ?? txt; } catch { /* keep text */ }
    if (res.status === 401) throw new UserError("OpenAI rejected the API key. Check OPENAI_API_KEY.", 502);
    if (res.status === 429) throw new UserError("OpenAI limit reached or no credit left: " + msg, 502);
    throw new UserError("OpenAI error: " + msg, 502);
  }
  const data = await res.json();
  let text = "";
  for (const item of data.output ?? []) {
    for (const c of item.content ?? []) {
      if (c.type === "output_text" && typeof c.text === "string") text += c.text;
      if (c.type === "refusal") throw new UserError("The AI declined to read this image.", 502);
    }
  }
  if (!text && typeof data.output_text === "string") text = data.output_text;
  try {
    return JSON.parse(text);
  } catch {
    throw new UserError("The AI returned an unreadable answer. Try again.", 502);
  }
}

function callOpenAI(payload: unknown) {
  return fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env("OPENAI_API_KEY")}`,
    },
    body: JSON.stringify(payload),
  });
}

// ------------------------------------------------------------- extraction

const QUESTION_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["questions"],
  properties: {
    questions: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: [
          "directions", "question_text", "question_type", "options", "answer",
          "answer_source", "explanation", "subject", "topic", "difficulty",
          "has_figure", "figure_box",
        ],
        properties: {
          directions: { type: "string" },
          question_text: { type: "string" },
          question_type: { type: "string", enum: ["mcq", "numeric", "text"] },
          options: {
            type: "array",
            items: {
              type: "object",
              additionalProperties: false,
              required: ["key", "text"],
              properties: { key: { type: "string" }, text: { type: "string" } },
            },
          },
          answer: { type: "string" },
          answer_source: { type: "string", enum: ["shown", "solved", "unknown"] },
          explanation: { type: "string" },
          subject: { type: "string" },
          topic: { type: "string" },
          difficulty: { type: "string", enum: ["easy", "medium", "hard"] },
          has_figure: { type: "boolean" },
          figure_box: {
            type: "object",
            additionalProperties: false,
            required: ["x", "y", "w", "h"],
            properties: {
              x: { type: "number" }, y: { type: "number" },
              w: { type: "number" }, h: { type: "number" },
            },
          },
        },
      },
    },
  },
};

function extractInstructions(subjects: string) {
  return `You read screenshots of banking-exam questions (IBPS, SBI, RRB mocks and papers) and turn them into clean question records.

Rules:
- Find every separate question in the image. Five questions means five records. Never merge them.
- A question cut off at the image edge with its statement or options missing: skip it.
- directions: shared instructions or passages for a set (a puzzle, a data table, a reading passage, "Directions (1-5)"). Copy the shared text into EVERY question of that set. Empty string if none.
- question_text: the full question statement only, without its number ("Q.12", "12.") and without the options.
- Copy text exactly. Do not fix, shorten or reword. If a word cannot be read, write [unclear].
- If text is in both English and Hindi, keep only the English.
- Maths: write expressions in LaTeX inside $...$, for example $\\frac{3}{4}$, $x^2$, $\\sqrt{144}$, $25\\%$. Plain numbers and words stay plain text.
- Tables in the image: write them as Markdown tables (| a | b |) inside directions or question_text.
- question_type: "mcq" when options are shown, "numeric" when a number is typed in, "text" otherwise.
- options: in screen order. key = the label shown (A, B, C... or 1, 2, 3...). If no labels are shown use A, B, C, D, E. Option text without its label.
- answer: for mcq the option key only (for example "C"); for numeric the number; for text the expected answer.
  answer_source "shown" if the image itself marks the correct answer (tick, green highlight, "Correct answer", answer key, solution).
  If not shown, solve the question carefully and use "solved". If you cannot solve it reliably, answer "" with "unknown".
- explanation: the solution shown in the image, copied. If none is shown and you solved it, a short worked solution. Otherwise "".
- subject: exactly one of: ${subjects}.
- topic: pick the best topic name from the list for that subject when one fits; otherwise suggest a short new topic name.
- difficulty: your estimate for a banking-exam candidate.
- has_figure: true only when the question needs a chart, graph, diagram, picture or a table that cannot be written as a simple Markdown table.
  figure_box: the figure's position as fractions of the image width and height (x, y = top-left corner, w, h = size, all between 0 and 1). Include titles, legends and axis labels. Use 0, 0, 0, 0 when has_figure is false.
- Return {"questions": []} when the image has no questions.`;
}

// deno-lint-ignore no-explicit-any
async function extract(sb: any, body: any) {
  const uploadId = String(body.upload_id ?? "");
  if (!uploadId) throw new UserError("upload_id is missing.");

  const { data: up, error: upErr } = await sb.from("uploads").select("*").eq("id", uploadId).single();
  if (upErr || !up) throw new UserError("Upload not found.", 404);
  if (up.status === "saved") throw new UserError("These questions were already saved.");

  const limit = parseInt(env("AI_DAILY_LIMIT", "40"), 10) || 40;
  const { data: ok, error: creditErr } = await sb.rpc("take_ai_credit", { p_limit: limit });
  if (creditErr) throw new UserError("Could not check the daily limit: " + creditErr.message, 500);
  if (!ok) throw new UserError(`Daily limit of ${limit} screenshots reached. It resets at midnight.`, 429);

  await sb.from("uploads").update({ status: "processing", error: null, updated_at: new Date().toISOString() })
    .eq("id", uploadId);

  try {
    const { data: file, error: dlErr } = await sb.storage.from(BUCKET).download(up.storage_path);
    if (dlErr || !file) throw new UserError("Could not read the screenshot from storage.", 500);
    const bytes = new Uint8Array(await file.arrayBuffer());
    const mime = file.type || "image/jpeg";
    const dataUrl = `data:${mime};base64,${toBase64(bytes)}`;

    const [{ data: subjects }, { data: topics }] = await Promise.all([
      sb.from("subjects").select("id,name,short_name").order("sort"),
      sb.from("topics").select("id,name,subject_id").order("sort"),
    ]);
    const subjList = (subjects ?? []).map((s: { name: string }) => `"${s.name}"`).join(", ");
    const topicLines = (subjects ?? []).map((s: { id: string; name: string }) =>
      `${s.name}: ${(topics ?? []).filter((t: { subject_id: string }) => t.subject_id === s.id)
        .map((t: { name: string }) => t.name).join("; ")}`
    ).join("\n");
    const hint = up.subject_hint
      ? (subjects ?? []).find((s: { id: string }) => s.id === up.subject_hint)?.name
      : null;

    const prompt = `Extract every question from this screenshot.
${hint ? `The user expects these to be ${hint} questions, but use the right subject for each question.\n` : ""}
Topics available per subject:
${topicLines}`;

    const result = await openai(
      env("OPENAI_MODEL", "gpt-6.2-sol"),
      [
        { type: "input_text", text: prompt },
        { type: "input_image", image_url: dataUrl, detail: "high" },
      ],
      "question_extraction",
      QUESTION_SCHEMA,
      extractInstructions(subjList),
    );

    const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
    // deno-lint-ignore no-explicit-any
    const questions = (result.questions ?? []).map((q: any, i: number) => {
      const subj = (subjects ?? []).find((s: { name: string; short_name: string }) =>
        norm(s.name) === norm(q.subject) || norm(s.short_name) === norm(q.subject))
        ?? (subjects ?? []).find((s: { id: string }) => s.id === up.subject_hint)
        ?? (subjects ?? [])[0];
      const topic = (topics ?? []).find((t: { subject_id: string; name: string }) =>
        t.subject_id === subj?.id && norm(t.name) === norm(q.topic));
      const box = q.figure_box ?? { x: 0, y: 0, w: 0, h: 0 };
      const clamp = (v: number) => Math.max(0, Math.min(1, Number(v) || 0));
      return {
        key: `${uploadId}:${i}`,
        upload_id: uploadId,
        directions: q.directions ?? "",
        question_text: q.question_text ?? "",
        qtype: ["mcq", "numeric", "text"].includes(q.question_type) ? q.question_type : "mcq",
        options: Array.isArray(q.options) ? q.options : [],
        answer: q.answer ?? "",
        answer_source: q.answer_source ?? "unknown",
        explanation: q.explanation ?? "",
        subject_id: subj?.id ?? null,
        topic_id: topic?.id ?? null,
        new_topic: topic ? "" : (q.topic ?? "").trim(),
        difficulty: { easy: 1, medium: 2, hard: 3 }[q.difficulty as string] ?? 2,
        has_figure: !!q.has_figure && clamp(box.w) > 0.02 && clamp(box.h) > 0.02,
        figure_box: { x: clamp(box.x), y: clamp(box.y), w: clamp(box.w), h: clamp(box.h) },
        screenshot_path: up.storage_path,
      };
    });

    await sb.from("uploads").update({
      status: "extracted", result: { questions }, error: null, updated_at: new Date().toISOString(),
    }).eq("id", uploadId);

    return { upload_id: uploadId, questions };
  } catch (e) {
    await sb.from("uploads").update({
      status: "failed", error: e instanceof Error ? e.message : String(e), updated_at: new Date().toISOString(),
    }).eq("id", uploadId);
    throw e;
  }
}

function toBase64(bytes: Uint8Array) {
  let s = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    s += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(s);
}

// ------------------------------------------------------ duplicate checks

const JUDGE_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["pairs"],
  properties: {
    pairs: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["id", "same"],
        properties: { id: { type: "integer" }, same: { type: "boolean" } },
      },
    },
  },
};

// deno-lint-ignore no-explicit-any
async function dedupe(sb: any, body: any) {
  const drafts = Array.isArray(body.drafts) ? body.drafts.slice(0, 40) : [];
  // deno-lint-ignore no-explicit-any
  const out: Record<string, any[]> = {};
  // deno-lint-ignore no-explicit-any
  const unsure: { id: number; key: string; draft: any; match: any }[] = [];

  for (const d of drafts) {
    const text = `${d.question_text ?? ""}`.trim();
    if (text.length < 12) continue;
    const { data, error } = await sb.rpc("find_similar", {
      p_text: text, p_subject: d.subject_id ?? null, p_exclude: d.exclude_id ?? null, p_limit: 4,
    });
    if (error) throw new UserError("Duplicate check failed: " + error.message, 500);
    for (const m of data ?? []) {
      const sameOptions = optionsKey(m.options) === optionsKey(d.options);
      if (m.exact || (m.similarity >= 0.97 && sameOptions)) {
        (out[d.key] ??= []).push({ ...m, verdict: "duplicate" });
      } else {
        unsure.push({ id: unsure.length, key: d.key, draft: d, match: m });
      }
    }
  }

  if (unsure.length) {
    const list = unsure.slice(0, 30).map((u) =>
      `Pair ${u.id}
NEW: ${clip(u.draft.directions)} ${clip(u.draft.question_text)} | options: ${optionsKey(u.draft.options)}
SAVED: ${clip(u.match.directions)} ${clip(u.match.question_text)} | options: ${optionsKey(u.match.options)}`
    ).join("\n\n");
    let verdicts: Record<number, boolean> = {};
    try {
      const r = await openai(
        env("OPENAI_DUP_MODEL", "gpt-6-luna"),
        [{ type: "input_text", text: list }],
        "duplicate_check",
        JUDGE_SCHEMA,
        `For each pair, decide if NEW and SAVED are the same exam question: the same problem asking for the same result, even if worded differently.
Different numbers, values, names or asked quantity mean a different question (same=false), because it is extra practice.
Return one entry per pair id.`,
      );
      for (const p of r.pairs ?? []) verdicts[p.id] = !!p.same;
    } catch {
      verdicts = {}; // AI unavailable: fall back to showing high-similarity matches
    }
    for (const u of unsure) {
      const v = verdicts[u.id];
      if (v === true || (v === undefined && u.match.similarity >= 0.8)) {
        (out[u.key] ??= []).push({ ...u.match, verdict: v === true ? "same problem" : "very similar" });
      }
    }
  }
  return { matches: out };
}

// deno-lint-ignore no-explicit-any
function optionsKey(opts: any) {
  if (!Array.isArray(opts)) return "";
  return opts.map((o) => `${o?.text ?? ""}`.toLowerCase().replace(/\s+/g, " ").trim()).join(" / ");
}
function clip(s: unknown, n = 600) {
  const t = `${s ?? ""}`.replace(/\s+/g, " ").trim();
  return t.length > n ? t.slice(0, n) + "..." : t;
}
