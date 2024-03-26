<template>
  <div class="mt-4 min-w-96">
    <div class="pg-right-content px-2">
      <ParameterPanel @layer-changed="$emit('layer-changed', $event)" />
      <!-- You can add more components as needed here -->
    </div>
  </div>
</template>

<script>
import CloseButton from './RightPanel/CloseButton.vue';
import ParameterPanel from './RightPanel/ParameterPanel.vue';

export default {
  components: {
    CloseButton,
    ParameterPanel
  },
  emits: ['layer-changed'],
  methods: {
    handleLayerChanged(layer) {
      // Handle layer change logic here
      console.log('Layer changed to', layer);
    }
  }
}
</script>
