<template>
  <div class="flex flex-col items-center justify-between">
    <div class="flex justify-between flex-row w-full mt-2">
      <label for="temperature-slider" class="text-gray-800 items-start">{{ name }}</label>
      <input type="text" v-model.number="value" @blur="formatValue" :step="step"
        class="w-14 border border-gray-300 rounded text-center text-sm focus:border-gray-800 focus:ring-gray-800" />
    </div>

    <div class="flex w-full items-center mt-2">
      <input id="temperature-slider" type="range" :min="min" :max="max" :step="step" v-model="value"
        @input="updateValue"
        class="appearance-none w-full h-2 bg-gray-200 rounded-lg cursor-pointer dark:bg-gray-500 accent-gray-700 border-black	" />

    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      value: 0.72,
    };
  },
  props: {
    name: String,
    min: Number,
    max: Number,
    step: {
      type: Number,
      default: 0.01,
    },
  },
  methods: {
    updateValue() {
      // Emit the value to the parent component or handle it as needed
      this.$emit('update:value', this.value);
    },
    formatValue() {
      // Ensure the value is a number and limit it to two decimal places
      this.value = parseFloat(this.value).toFixed(2);
    },
  },
  watch: {
    value(newValue, oldValue) {
      if (newValue !== oldValue) {
        this.updateValue();
      }
    },
  },
};
</script>
