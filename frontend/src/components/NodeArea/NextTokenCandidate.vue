<template>
  <div class="ring-1 p-2">
    <div class="flex flex-col">
      <div class="flex flex-row items-center mt-2" v-for="(token, index) in tokens" :key="index">
        <Token :tokenId="indices[index]" :text="token" :index="index" :activation="logits[index]" />
        <div>{{ logits[index].toFixed(2) }}</div>
      </div>
    </div>
  </div>
</template>
<script>
import Token from './Token.vue';

export default {
  components: {
    Token
  },
  props: {
    tokens: {
      type: Array,
      required: true
    },
    logits: {
      type: Array,
      required: true
    },
    indices: {
      type: Array,
      required: true
    }
  }
};
</script>
