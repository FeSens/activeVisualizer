<template>
  <div class="flex flex-col w-full">
    <TextArea :token-count="tokenCount" @text-changed="text = $event" />
    <div class="flex flex-col ml-2 mt-2">
      <h1>Next Token Candidate</h1>
      <NextTokenCandidate :tokens="lastToken" :logits="lastLogit" :indices="lastIndices" />
    </div>
    <div v-for="(head, index) in activation" :key="head" class="flex flex-col mt-2 ml-2 ring-1 p-2">
      <p class="text-sm font-bold text-gray-500">{{ layer }}_{{ index }}</p>
      <div class="flex flex-row flex-wrap">
        <Token v-for="token in tokens" :key="`${token.tokenId}${index}`" :token-id="token.tokenId" :text="token.text"
          :index="token.index" :activation="head[activeToken][token.index]" @hovered="handleHovered" class="mt-2" />
      </div>
    </div>
  </div>
</template>

<script>
import TextArea from './NodeArea/TextArea.vue';
import NextTokenCandidate from './NodeArea/NextTokenCandidate.vue';
import Token from './NodeArea/Token.vue';
import useModel from '../composables/useModel';
import { ref, reactive, watch } from 'vue';

export default {
  components: {
    TextArea,
    Token,
    NextTokenCandidate
  },
  props: {
    layer: {
      type: String,
      default: 'transformer.h.0.attn.softmax.0'
    }
  },
  setup(props) {
    const {
      tokens,
      tokenCount,
      activation,
      text,
      layer,
      logits,
      logitsTokens,
      logitsIndices,
      isModelReady,
    } = useModel();

    watch(() => props.layer, (newLayer) => {
      layer.value = newLayer;
    });
    const activeToken = ref(0);

    return {
      tokens,
      tokenCount,
      activation,
      text,
      isModelReady,
      logits,
      logitsTokens,
      logitsIndices,
      activeToken
    }
  },
  computed: {
    lastToken() {
      return this.logitsTokens[this.logitsTokens.length - 1];
    },
    lastLogit() {
      return this.logits[this.logits.length - 1];
    },
    lastIndices() {
      return this.logitsIndices[this.logitsIndices.length - 1];
    }
  },
  watch: {
    logitsTokens(newActiveToken) {
      console.log(this.logitsTokens)
    }
  },
  methods: {
    handleHovered(data) {
      this.activeToken = data.index;
    }
  }
}
</script>