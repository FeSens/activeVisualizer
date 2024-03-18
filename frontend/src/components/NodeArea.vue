<template>
  <div class="flex flex-col w-full">
      <TextArea :token-count="tokenCount" @text-changed="handleTextChanged" />

      <div v-for="head in activation" :key="head" class="flex flex-row flex-wrap mt-2">
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
import useWebSockets from '../composables/baseSockets';
import { ref } from 'vue';

export default {
  components: {
    TextArea,
    Token
  },
  setup() {
    const { tokenizeText, getActivations } = useWebSockets();
    const { isOpen, data, send } = tokenizeText;
    const { data: activations, send: forward } = getActivations;
    const tokenCount = ref(0);
    const activation = ref([[]]);
    const activeToken = ref(0);
    const layer = ref('model.layers.20.self_attn.softmax.0');
    return {
      tokenCount,
      isOpen,
      data,
      send,
      activations,
      forward,
      activation,
      activeToken,
      layer
    }
  },
  methods: {
    handleTextChanged(text) {
      this.send(text);
      this.forward(JSON.stringify({
        layer_name: this.layer,
        text,
        top_k: 3,
      }));
    },
    handleHovered(data) {
      this.activeToken = data.index;
    }
  },
  watch: {
    data(newData) {
      const parsedData = JSON.parse(newData);
      this.tokenCount = parsedData.tokens.length;
    },
    activations(newActivations) {
      const parsedActivations = JSON.parse(newActivations);
      const capturedTargets = parsedActivations["captured_targets"][`${this.layer}.activ_norm`][0];
      this.activation = capturedTargets;
      // let head_1 = capturedTargets[0][0];
      // Transpose the matrix
      // head_1 = head_1[0].map((_, colIndex) => head_1.map(row => row[colIndex]));
      // normalize the min and max values to 0 and 1, for each of the lines (x, y)
      // head_1 = head_1.map((line) => {
      //   const min = Math.min(...line);
      //   const max = Math.max(...line);
      //   return line.map((value) => (value - min) / (max - min));
      // });
      // this.activation = head_1;
      // console.log(head_1);
      
    }
  },
  computed: {
    tokens() {
      if (!this.data) return [];
      const parsedData = JSON.parse(this.data);
      console.log(parsedData);
      return parsedData.text.map((t, index) => ({
        text: t,
        tokenId: parsedData.tokens[index],
        index,
      }));
    },
  },
}
</script>