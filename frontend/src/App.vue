<template>
  <div class="flex flex-row-reverse h-screen mt-2 ml-2">
    <RightPanel @layer-changed="handleLayerChange" />
    <NodeArea :layer="layer" />
  </div>
</template>

<script>
import RightPanel from './components/RightPanel.vue';
import NodeArea from './components/NodeArea.vue';
import { ref } from 'vue';

export default {
  components: {
    RightPanel,
    NodeArea
  },
  setup() {
    const layer = ref('');
    return {
      layer
    }
  },
  methods: {
    handleLayerChange(layer) {
      this.layer = layer;
    }
  }
}
</script>