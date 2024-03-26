
import baseSockets from "../api/baseSockets";
import { ref, reactive, computed, watch } from 'vue';

export default function useModel() {
  const { isOpen: tokenWebsocketIsOpen, data: tokenData, send: tokenSend } = baseSockets.tokenizeText;
  const { isOpen: inferenceWebsocketIsOpen, data: activations, send: forward } = baseSockets.getActivations;
  const tokenCount = ref(0);
  const activation = ref([[[]]]);
  const logits = ref([]);
  const logitsTokens = ref([]);
  const logitsIndices = ref([]);
  const text = ref('');
  const layer = ref('');
  const attentionHeadsAblated = reactive([]);

  watch([text, layer, attentionHeadsAblated], ([newText, newLayer, newAttentionHeadsAblated], [prevText, prevLayer, prevAttentionHeadsAblated]) => {
    if (newText !== prevText) {
      tokenSend(newText);
    }
    if (newText === '' || newLayer === '') return;
    forward(JSON.stringify({
      layer_name: newLayer,
      text: newText,
      top_k: 3,
      positions_to_ablate: newAttentionHeadsAblated
      // positions_to_ablate: [
      //   // [2, 0.0, 'transformer.h.11.attn.matmul.1'], [4, 0.0, 'transformer.h.11.attn.matmul.1'], [5, 0.0, 'transformer.h.11.attn.matmul.1'], [6, 0.0, 'transformer.h.11.attn.matmul.1'], [7, 0.0, 'transformer.h.11.attn.matmul.1'], [1, 0.0, 'transformer.h.5.attn.matmul.1'], [1, 0.0, 'transformer.h.5.attn.matmul.1'], [9, 0.0, 'transformer.h.6.attn.matmul.1'], [9, 0.0, 'transformer.h.6.attn.matmul.1'], [1, 0.0, 'transformer.h.7.attn.matmul.1'], [2, 0.0, 'transformer.h.7.attn.matmul.1'], [10, 0.0, 'transformer.h.7.attn.matmul.1'], [1, 0.0, 'transformer.h.7.attn.matmul.1'], [2, 0.0, 'transformer.h.7.attn.matmul.1'], [10, 0.0, 'transformer.h.7.attn.matmul.1'], [1, 0.0, 'transformer.h.8.attn.matmul.1'], [1, 0.0, 'transformer.h.8.attn.matmul.1'], [1, 0.0, 'transformer.h.9.attn.matmul.1'], [1, 0.0, 'transformer.h.9.attn.matmul.1'], [8, 0.0, 'transformer.h.10.attn.matmul.1'], [8, 0.0, 'transformer.h.10.attn.matmul.1'],
      //   // [11, 0.0, 'transformer.h.2.attn.matmul.1'], [11, 0.0, 'transformer.h.2.attn.matmul.1'], [4, 0.0, 'transformer.h.3.attn.matmul.1'], [4, 0.0, 'transformer.h.3.attn.matmul.1'], [2, 0.0, 'transformer.h.4.attn.matmul.1'], [4, 0.0, 'transformer.h.4.attn.matmul.1'], [8, 0.0, 'transformer.h.4.attn.matmul.1'], [10, 0.0, 'transformer.h.4.attn.matmul.1'], [2, 0.0, 'transformer.h.4.attn.matmul.1'], [4, 0.0, 'transformer.h.4.attn.matmul.1'], [8, 0.0, 'transformer.h.4.attn.matmul.1'], [10, 0.0, 'transformer.h.4.attn.matmul.1'], [1, 0.0, 'transformer.h.5.attn.matmul.1'], [1, 0.0, 'transformer.h.5.attn.matmul.1'], [2, 0.0, 'transformer.h.6.attn.matmul.1'], [5, 0.0, 'transformer.h.6.attn.matmul.1'], [9, 0.0, 'transformer.h.6.attn.matmul.1'], [10, 0.0, 'transformer.h.6.attn.matmul.1'], [2, 0.0, 'transformer.h.6.attn.matmul.1'], [5, 0.0, 'transformer.h.6.attn.matmul.1'], [9, 0.0, 'transformer.h.6.attn.matmul.1'], [10, 0.0, 'transformer.h.6.attn.matmul.1'], [1, 0.0, 'transformer.h.7.attn.matmul.1'], [2, 0.0, 'transformer.h.7.attn.matmul.1'], [4, 0.0, 'transformer.h.7.attn.matmul.1'], [5, 0.0, 'transformer.h.7.attn.matmul.1'], [6, 0.0, 'transformer.h.7.attn.matmul.1'], [7, 0.0, 'transformer.h.7.attn.matmul.1'], [10, 0.0, 'transformer.h.7.attn.matmul.1'], [11, 0.0, 'transformer.h.7.attn.matmul.1'], [1, 0.0, 'transformer.h.7.attn.matmul.1'], [2, 0.0, 'transformer.h.7.attn.matmul.1'], [4, 0.0, 'transformer.h.7.attn.matmul.1'], [5, 0.0, 'transformer.h.7.attn.matmul.1'], [6, 0.0, 'transformer.h.7.attn.matmul.1'], [7, 0.0, 'transformer.h.7.attn.matmul.1'], [10, 0.0, 'transformer.h.7.attn.matmul.1'], [11, 0.0, 'transformer.h.7.attn.matmul.1'], [0, 0.0, 'transformer.h.8.attn.matmul.1'], [1, 0.0, 'transformer.h.8.attn.matmul.1'], [9, 0.0, 'transformer.h.8.attn.matmul.1'], [0, 0.0, 'transformer.h.8.attn.matmul.1'], [1, 0.0, 'transformer.h.8.attn.matmul.1'], [9, 0.0, 'transformer.h.8.attn.matmul.1'], [0, 0.0, 'transformer.h.9.attn.matmul.1'], [1, 0.0, 'transformer.h.9.attn.matmul.1'], [2, 0.0, 'transformer.h.9.attn.matmul.1'], [4, 0.0, 'transformer.h.9.attn.matmul.1'], [5, 0.0, 'transformer.h.9.attn.matmul.1'], [11, 0.0, 'transformer.h.9.attn.matmul.1'], [0, 0.0, 'transformer.h.9.attn.matmul.1'], [1, 0.0, 'transformer.h.9.attn.matmul.1'], [2, 0.0, 'transformer.h.9.attn.matmul.1'], [4, 0.0, 'transformer.h.9.attn.matmul.1'], [5, 0.0, 'transformer.h.9.attn.matmul.1'], [11, 0.0, 'transformer.h.9.attn.matmul.1'], [1, 0.0, 'transformer.h.10.attn.matmul.1'], [3, 0.0, 'transformer.h.10.attn.matmul.1'], [8, 0.0, 'transformer.h.10.attn.matmul.1'], [1, 0.0, 'transformer.h.10.attn.matmul.1'], [3, 0.0, 'transformer.h.10.attn.matmul.1'], [8, 0.0, 'transformer.h.10.attn.matmul.1'], [2, 0.0, 'transformer.h.11.attn.matmul.1'], [4, 0.0, 'transformer.h.11.attn.matmul.1'], [5, 0.0, 'transformer.h.11.attn.matmul.1'], [6, 0.0, 'transformer.h.11.attn.matmul.1'], [7, 0.0, 'transformer.h.11.attn.matmul.1'], [9, 0.0, 'transformer.h.11.attn.matmul.1'], [2, 0.0, 'transformer.h.11.attn.matmul.1'], [4, 0.0, 'transformer.h.11.attn.matmul.1'], [5, 0.0, 'transformer.h.11.attn.matmul.1'], [6, 0.0, 'transformer.h.11.attn.matmul.1'], [7, 0.0, 'transformer.h.11.attn.matmul.1'], [9, 0.0, 'transformer.h.11.attn.matmul.1'],
      //   // [0, 0.0, 'transformer.h.0.attn.matmul.1'], [9, 0.0, 'transformer.h.0.attn.matmul.1'], [11, 0.0, 'transformer.h.0.attn.matmul.1'], [0, 0.0, 'transformer.h.0.attn.matmul.1'], [9, 0.0, 'transformer.h.0.attn.matmul.1'], [11, 0.0, 'transformer.h.0.attn.matmul.1'], [3, 0.0, 'transformer.h.1.attn.matmul.1'], [4, 0.0, 'transformer.h.1.attn.matmul.1'], [6, 0.0, 'transformer.h.1.attn.matmul.1'], [7, 0.0, 'transformer.h.1.attn.matmul.1'], [8, 0.0, 'transformer.h.1.attn.matmul.1'], [9, 0.0, 'transformer.h.1.attn.matmul.1'], [3, 0.0, 'transformer.h.1.attn.matmul.1'], [4, 0.0, 'transformer.h.1.attn.matmul.1'], [6, 0.0, 'transformer.h.1.attn.matmul.1'], [7, 0.0, 'transformer.h.1.attn.matmul.1'], [8, 0.0, 'transformer.h.1.attn.matmul.1'], [9, 0.0, 'transformer.h.1.attn.matmul.1'], [1, 0.0, 'transformer.h.2.attn.matmul.1'], [6, 0.0, 'transformer.h.2.attn.matmul.1'], [11, 0.0, 'transformer.h.2.attn.matmul.1'], [1, 0.0, 'transformer.h.2.attn.matmul.1'], [6, 0.0, 'transformer.h.2.attn.matmul.1'], [11, 0.0, 'transformer.h.2.attn.matmul.1'], [4, 0.0, 'transformer.h.3.attn.matmul.1'], [5, 0.0, 'transformer.h.3.attn.matmul.1'], [10, 0.0, 'transformer.h.3.attn.matmul.1'], [4, 0.0, 'transformer.h.3.attn.matmul.1'], [5, 0.0, 'transformer.h.3.attn.matmul.1'], [10, 0.0, 'transformer.h.3.attn.matmul.1'], [2, 0.0, 'transformer.h.4.attn.matmul.1'], [4, 0.0, 'transformer.h.4.attn.matmul.1'], [5, 0.0, 'transformer.h.4.attn.matmul.1'], [6, 0.0, 'transformer.h.4.attn.matmul.1'], [8, 0.0, 'transformer.h.4.attn.matmul.1'], [10, 0.0, 'transformer.h.4.attn.matmul.1'], [2, 0.0, 'transformer.h.4.attn.matmul.1'], [4, 0.0, 'transformer.h.4.attn.matmul.1'], [5, 0.0, 'transformer.h.4.attn.matmul.1'], [6, 0.0, 'transformer.h.4.attn.matmul.1'], [8, 0.0, 'transformer.h.4.attn.matmul.1'], [10, 0.0, 'transformer.h.4.attn.matmul.1'], [1, 0.0, 'transformer.h.5.attn.matmul.1'], [3, 0.0, 'transformer.h.5.attn.matmul.1'], [11, 0.0, 'transformer.h.5.attn.matmul.1'], [1, 0.0, 'transformer.h.5.attn.matmul.1'], [3, 0.0, 'transformer.h.5.attn.matmul.1'], [11, 0.0, 'transformer.h.5.attn.matmul.1'], [2, 0.0, 'transformer.h.6.attn.matmul.1'], [5, 0.0, 'transformer.h.6.attn.matmul.1'], [6, 0.0, 'transformer.h.6.attn.matmul.1'], [9, 0.0, 'transformer.h.6.attn.matmul.1'], [10, 0.0, 'transformer.h.6.attn.matmul.1'], [11, 0.0, 'transformer.h.6.attn.matmul.1'], [2, 0.0, 'transformer.h.6.attn.matmul.1'], [5, 0.0, 'transformer.h.6.attn.matmul.1'], [6, 0.0, 'transformer.h.6.attn.matmul.1'], [9, 0.0, 'transformer.h.6.attn.matmul.1'], [10, 0.0, 'transformer.h.6.attn.matmul.1'], [11, 0.0, 'transformer.h.6.attn.matmul.1'], [0, 0.0, 'transformer.h.7.attn.matmul.1'], [1, 0.0, 'transformer.h.7.attn.matmul.1'], [2, 0.0, 'transformer.h.7.attn.matmul.1'], [4, 0.0, 'transformer.h.7.attn.matmul.1'], [5, 0.0, 'transformer.h.7.attn.matmul.1'], [6, 0.0, 'transformer.h.7.attn.matmul.1'], [7, 0.0, 'transformer.h.7.attn.matmul.1'], [10, 0.0, 'transformer.h.7.attn.matmul.1'], [11, 0.0, 'transformer.h.7.attn.matmul.1'], [0, 0.0, 'transformer.h.7.attn.matmul.1'], [1, 0.0, 'transformer.h.7.attn.matmul.1'], [2, 0.0, 'transformer.h.7.attn.matmul.1'], [4, 0.0, 'transformer.h.7.attn.matmul.1'], [5, 0.0, 'transformer.h.7.attn.matmul.1'], [6, 0.0, 'transformer.h.7.attn.matmul.1'], [7, 0.0, 'transformer.h.7.attn.matmul.1'], [10, 0.0, 'transformer.h.7.attn.matmul.1'], [11, 0.0, 'transformer.h.7.attn.matmul.1'], [0, 0.0, 'transformer.h.8.attn.matmul.1'], [1, 0.0, 'transformer.h.8.attn.matmul.1'], [2, 0.0, 'transformer.h.8.attn.matmul.1'], [8, 0.0, 'transformer.h.8.attn.matmul.1'], [9, 0.0, 'transformer.h.8.attn.matmul.1'], [11, 0.0, 'transformer.h.8.attn.matmul.1'], [0, 0.0, 'transformer.h.8.attn.matmul.1'], [1, 0.0, 'transformer.h.8.attn.matmul.1'], [2, 0.0, 'transformer.h.8.attn.matmul.1'], [8, 0.0, 'transformer.h.8.attn.matmul.1'], [9, 0.0, 'transformer.h.8.attn.matmul.1'], [11, 0.0, 'transformer.h.8.attn.matmul.1'], [0, 0.0, 'transformer.h.9.attn.matmul.1'], [1, 0.0, 'transformer.h.9.attn.matmul.1'], [2, 0.0, 'transformer.h.9.attn.matmul.1'], [4, 0.0, 'transformer.h.9.attn.matmul.1'], [5, 0.0, 'transformer.h.9.attn.matmul.1'], [7, 0.0, 'transformer.h.9.attn.matmul.1'], [8, 0.0, 'transformer.h.9.attn.matmul.1'], [11, 0.0, 'transformer.h.9.attn.matmul.1'], [0, 0.0, 'transformer.h.9.attn.matmul.1'], [1, 0.0, 'transformer.h.9.attn.matmul.1'], [2, 0.0, 'transformer.h.9.attn.matmul.1'], [4, 0.0, 'transformer.h.9.attn.matmul.1'], [5, 0.0, 'transformer.h.9.attn.matmul.1'], [7, 0.0, 'transformer.h.9.attn.matmul.1'], [8, 0.0, 'transformer.h.9.attn.matmul.1'], [11, 0.0, 'transformer.h.9.attn.matmul.1'], [1, 0.0, 'transformer.h.10.attn.matmul.1'], [2, 0.0, 'transformer.h.10.attn.matmul.1'], [3, 0.0, 'transformer.h.10.attn.matmul.1'], [4, 0.0, 'transformer.h.10.attn.matmul.1'], [6, 0.0, 'transformer.h.10.attn.matmul.1'], [8, 0.0, 'transformer.h.10.attn.matmul.1'], [1, 0.0, 'transformer.h.10.attn.matmul.1'], [2, 0.0, 'transformer.h.10.attn.matmul.1'], [3, 0.0, 'transformer.h.10.attn.matmul.1'], [4, 0.0, 'transformer.h.10.attn.matmul.1'], [6, 0.0, 'transformer.h.10.attn.matmul.1'], [8, 0.0, 'transformer.h.10.attn.matmul.1'], [1, 0.0, 'transformer.h.11.attn.matmul.1'], [2, 0.0, 'transformer.h.11.attn.matmul.1'], [4, 0.0, 'transformer.h.11.attn.matmul.1'], [5, 0.0, 'transformer.h.11.attn.matmul.1'], [6, 0.0, 'transformer.h.11.attn.matmul.1'], [7, 0.0, 'transformer.h.11.attn.matmul.1'], [9, 0.0, 'transformer.h.11.attn.matmul.1'], [1, 0.0, 'transformer.h.11.attn.matmul.1'], [2, 0.0, 'transformer.h.11.attn.matmul.1'], [4, 0.0, 'transformer.h.11.attn.matmul.1'], [5, 0.0, 'transformer.h.11.attn.matmul.1'], [6, 0.0, 'transformer.h.11.attn.matmul.1'], [7, 0.0, 'transformer.h.11.attn.matmul.1'], [9, 0.0, 'transformer.h.11.attn.matmul.1'],
      //   // [0, 0.0, 'transformer.h.10.attn.matmul.1'], [1, 0.0, 'transformer.h.10.attn.matmul.1'], [2, 0.0, 'transformer.h.10.attn.matmul.1'], [3, 0.0, 'transformer.h.10.attn.matmul.1'], [4, 0.0, 'transformer.h.10.attn.matmul.1'], [6, 0.0, 'transformer.h.10.attn.matmul.1'], [8, 0.0, 'transformer.h.10.attn.matmul.1'], [10, 0.0, 'transformer.h.10.attn.matmul.1'], [0, 0.0, 'transformer.h.10.attn.matmul.1'], [1, 0.0, 'transformer.h.10.attn.matmul.1'], [2, 0.0, 'transformer.h.10.attn.matmul.1'], [3, 0.0, 'transformer.h.10.attn.matmul.1'], [4, 0.0, 'transformer.h.10.attn.matmul.1'], [6, 0.0, 'transformer.h.10.attn.matmul.1'], [8, 0.0, 'transformer.h.10.attn.matmul.1'], [10, 0.0, 'transformer.h.10.attn.matmul.1'],
      //   // [1, 0.0, 'transformer.h.10.attn.matmul.1'], [3, 0.0, 'transformer.h.10.attn.matmul.1'], [6, 0.0, 'transformer.h.10.attn.matmul.1'], [8, 0.0, 'transformer.h.10.attn.matmul.1'], [1, 0.0, 'transformer.h.10.attn.matmul.1'], [3, 0.0, 'transformer.h.10.attn.matmul.1'], [6, 0.0, 'transformer.h.10.attn.matmul.1'], [8, 0.0, 'transformer.h.10.attn.matmul.1'], [2, 0.0, 'transformer.h.11.attn.matmul.1'], [4, 0.0, 'transformer.h.11.attn.matmul.1'], [5, 0.0, 'transformer.h.11.attn.matmul.1'], [6, 0.0, 'transformer.h.11.attn.matmul.1'], [7, 0.0, 'transformer.h.11.attn.matmul.1'], [9, 0.0, 'transformer.h.11.attn.matmul.1'], [2, 0.0, 'transformer.h.11.attn.matmul.1'], [4, 0.0, 'transformer.h.11.attn.matmul.1'], [5, 0.0, 'transformer.h.11.attn.matmul.1'], [6, 0.0, 'transformer.h.11.attn.matmul.1'], [7, 0.0, 'transformer.h.11.attn.matmul.1'], [9, 0.0, 'transformer.h.11.attn.matmul.1'],
      // ]
    }));
  });

  watch(tokenData, (newData) => {
    const parsedData = JSON.parse(newData);
    tokenCount.value = parsedData.tokens.length;
  });

  watch(activations, (newActivations) => {
    const parsedActivations = JSON.parse(newActivations);
    const capturedTargets = parsedActivations["captured_targets"][`${layer.value}.activ_norm`][0];
    activation.value = capturedTargets;
    logits.value = parsedActivations["logits_values"];
    logitsTokens.value = parsedActivations["logits_tokens"];
    logitsIndices.value = parsedActivations["logits_indices"];
  });

  const tokens = computed(() => {
    if (!tokenData.value) return [];
    const parsedData = JSON.parse(tokenData.value);
    return parsedData.text.map((t: string, index: number) => ({
      text: t,
      tokenId: parsedData.tokens[index],
      index,
    }));
  });

  const isModelReady = computed(() => {
    return tokenWebsocketIsOpen && inferenceWebsocketIsOpen;
  });

  return {
    tokens,
    tokenCount,
    activation,
    text,
    layer,
    logits,
    logitsTokens,
    logitsIndices,
    isModelReady,
    attentionHeadsAblated,
  }

}