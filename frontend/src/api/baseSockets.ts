/* eslint-disable @typescript-eslint/no-non-null-assertion */
/* eslint-disable @typescript-eslint/explicit-module-boundary-types */
import { useWebSocket } from "vue-composable";

export default {
  tokenizeText: useWebSocket(
    "ws://localhost:8000/ws/tokenizer",
  ),
  getActivations: useWebSocket(
    "ws://localhost:8000/ws/model/forward",
  ),
};
