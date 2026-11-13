import OpenAI from "openai";

let client: OpenAI | undefined;

export function getOpenAIClient(): OpenAI {
  return client ??= new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
}
