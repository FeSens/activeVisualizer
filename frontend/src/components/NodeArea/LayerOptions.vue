<template>
  <div class="relative inline-block">
    <button @click="toggleLayerOptions" class="text-sm text-gray-600 focus:outline-none">
      <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
          d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
      </svg>
    </button>
    <div v-if="showLayerOptions"
      class="absolute right-0 mt-2 w-40 bg-white border border-gray-300 rounded-md shadow-lg z-10">
      <div class="flex flex-col space-y-2 p-4">
        <div class="flex flex-row justify-between items-center">
          <button @click="handleAblate"
            class="text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900">
            {{ isHeadAblated() ? 'Unablate' : 'Ablate' }}
          </button>
          <input v-if="!isHeadAblated()" v-model="value" type="number" step="0.1"
            class="w-16 px-2 py-1 border border-gray-300 rounded-md">
        </div>
        <button @click="handleTraceCircuits"
          class="text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900">
          Trace Circuits
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';

const props = defineProps({
  layer: {
    type: String,
    required: true,
  },
  index: {
    type: Number,
    required: true,
  }
});

const attentionHeadsAblated = defineModel('attentionHeadsAblated');
const showLayerOptions = ref(false);
const value = ref(0);

const toggleLayerOptions = () => {
  showLayerOptions.value = !showLayerOptions.value;
};

const isHeadAblated = () => {
  return attentionHeadsAblated.value.some(h => h[0] === props.index && h[2] === props.layer);
};

const handleAblate = () => {
  const head = [props.index, value.value, props.layer];
  const index = attentionHeadsAblated.value.findIndex(h => h[0] === props.index && h[2] === props.layer);
  if (index !== -1) {
    attentionHeadsAblated.value.splice(index, 1);
  } else {
    attentionHeadsAblated.value.push(head);
  }
  showLayerOptions.value = false;
};

const handleTraceCircuits = () => {
  console.log('Trace Circuits option selected');
  showLayerOptions.value = false;
};

const closeLayerOptions = (event) => {
  if (!event.target.closest('.relative.inline-block')) {
    showLayerOptions.value = false;
  }
};

onMounted(() => {
  document.addEventListener('click', closeLayerOptions);
});

onUnmounted(() => {
  document.removeEventListener('click', closeLayerOptions);
});
</script>