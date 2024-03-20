<template>
  <div class="flex flex-col w-full">
      <TextArea :token-count="tokenCount" @text-changed="text=$event" />

      <div v-for="(head, index) in activation" :key="head" class="flex flex-col flex-wrap mt-2 ml-2 ring-1 p-2">
        <p class="text-sm font-bold text-gray-500">{{ layer }}_{{ index }}</p>
        <div class="flex flex-row">
          <Token
            v-for="token in tokens"
            :key="token.tokenId"
            :token-id="token.tokenId"
            :text="token.text"
            :index="token.index"
            :activation="head[activeToken][token.index]"
            @hovered="handleHovered"
          />
        </div>
      </div>
      
      <!-- <div class="flex flex-row flex-wrap mt-2">
        <Token
          v-for="token in tokens"
          :key="token.tokenId"
          :token-id="token.tokenId"
          :text="token.text"
          :index="token.index"
          :activation="activation[activeToken][token.index]"
          @hovered="handleHovered"
        />
      </div> -->
  </div>
</template>

<script>
import TextArea from './NodeArea/TextArea.vue';
import Token from './NodeArea/Token.vue';
import useModel from '../composables/useModel';
import { ref, reactive, watch } from 'vue';

export default {
  components: {
    TextArea,
    Token
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
      activeToken
    }
  },
  methods: {
    handleHovered(data) {
      this.activeToken = data.index;
    }
  }
}
</script>