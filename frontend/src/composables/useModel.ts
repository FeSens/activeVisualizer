
import baseSockets from "../api/baseSockets";
import { ref, Ref, computed, watch } from 'vue';

export default function useModel() {
  const { isOpen: tokenWebsocketIsOpen, data: tokenData, send: tokenSend } = baseSockets.tokenizeText;
  const { isOpen: inferenceWebsocketIsOpen, data: activations, send: forward } = baseSockets.getActivations;
  const tokenCount = ref(0);
  const activation = ref([[]]);
  const text = ref('');
  const layer = ref('');

  watch([text, layer], ([newText, newLayer], [prevText, _]) => {
    if (newText !== prevText) {
      tokenSend(newText);
    }
    if (newText === '' || newLayer === '') return;
    forward(JSON.stringify({
      layer_name: newLayer,
      text: newText,
      top_k: 3,
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
  });

  const tokens = computed(() => {
    if (!tokenData.value) return [];
    const parsedData = JSON.parse(tokenData.value);
    return parsedData.text.map((t, index) => ({
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
    isModelReady,
  }

}