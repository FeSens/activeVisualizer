<template>
  <div class="parameter-panel">
    <div class="parameter-panel-grid">
      <!-- Simplified example, replace with actual components as needed -->
      <!-- <DropdownSelect placeholder="Select Model" />
      <SliderControl name="Temperature" :min="0" :max="2" />
      <SliderControl name="Max Length" :min="0" :max="1024" :step="1" /> -->
      <!-- Make a division here of some sort -->
      <DropdownSelect placeholder="Select Layer" :people="layers" @selected-person-changed="$emit('layer-changed', $event)"/>
      <!-- <SliderControl name="Top P" :min="0" :max="32" :step="1" /> -->
      <!-- Add more controls as needed -->
    </div>
  </div>
</template>

<script>
import DropdownSelect from './DropdownSelect.vue';
import SliderControl from './SliderControl.vue';
import useLayers from '../../composables/useLayers';


export default {
  components: {
    DropdownSelect,
    SliderControl
  },
  emits: ['layer-changed'],
  setup() {
    const { layers, getLayers } = useLayers();
    getLayers();
    return {
      layers
    }
  }
}
</script>
