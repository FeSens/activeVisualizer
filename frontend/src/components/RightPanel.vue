<template>
  <div class="mt-4 w-64">
    <div class="pg-right-content px-2">
      <CloseButton @close="handleClose" />
      <ParameterPanel />
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
  methods: {
    handleClose() {
      // Close panel logic here
      console.log('Closing panel');
    }
  }
}
</script>
