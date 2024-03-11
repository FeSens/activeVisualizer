<template>
  <button class="pg-right-panel-mobile-close" @click="close">×</button>
</template>

<script>
export default {
  emits: ['close'],
  methods: {
    close() {
      this.$emit('close');
    }
  }
}
</script>
