import { initLogger } from "braintrust";

export const logger = initLogger({
  projectName: "Red Panda Simple",
  apiKey: process.env.BRAINTRUST_API_KEY,
});