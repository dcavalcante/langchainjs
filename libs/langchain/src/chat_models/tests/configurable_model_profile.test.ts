import { describe, expect, it, vi } from "vitest";
import { RunnableBinding, RunnableLambda } from "@langchain/core/runnables";
import { initChatModel } from "../universal.js";

class DummyChatModel {
  profile = { maxInputTokens: 999 };

  constructor(_fields: { model?: string }) {}

  bindTools() {
    return new RunnableBinding({
      bound: RunnableLambda.from((input: unknown) => input),
      config: {},
    });
  }
}

vi.mock("@langchain/openai", () => ({ ChatOpenAI: DummyChatModel }));

describe("ConfigurableModel profile preservation", () => {
  it("preserves an explicit profile in the original and tool-bound model", async () => {
    const model = await initChatModel("gpt-4o-mini", {
      profile: { maxInputTokens: 100000 },
    });
    expect(model.profile.maxInputTokens).toBe(100000);

    const toolBoundModel = model.bindTools([]);
    const initializedModel = await toolBoundModel._getModelInstance();
    expect(initializedModel).toBeInstanceOf(RunnableBinding);
    expect("profile" in initializedModel).toBe(false);

    expect(toolBoundModel.profile.maxInputTokens).toBe(100000);
  });

  it("continues to infer a profile when no explicit override is supplied", async () => {
    const model = await initChatModel("gpt-4o-mini");
    expect(model.profile.maxInputTokens).toBe(999);
  });
});
