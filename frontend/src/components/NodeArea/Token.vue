<template>
  <div :id="tokenId" :style="backgroundColorStyle" class="rounded mr-1 relative pt-2 min-w-7"
    @mouseover="emitHoverEvent">
    <span class="token-index absolute top-0 left-1 text-xs" :data-token-id="tokenId">{{ index }}.</span>
    <span class="token pl-3 pr-1" :data-token-id="tokenId">{{ text.replace(" ", "_").replace('\n', '\\n') }}</span>
  </div>
</template>

<script>
export default {
  emits: ['hovered'],
  props: {
    tokenId: {
      type: Number,
      required: true,
    },
    text: {
      type: String,
      required: true,
    },
    index: {
      type: Number,
      required: true,
    },
    activation: {
      type: Number,
      required: false,
      default: 0,
    },
  },
  methods: {
    emitHoverEvent() {
      // Emit a custom event named 'hovered' when the mouse hovers over the component
      // You can pass any data as the second argument of $emit, such as the tokenId or text
      this.$emit('hovered', { tokenId: this.tokenId, text: this.text, index: this.index });
    },
  },
  computed: {
    backgroundColorStyle() {
      // Assuming gray is simply low saturation, and green is at 120 degrees in HSL
      // Convert activation to a saturation value for HSL
      let saturation = this.activation * 100; // Scale activation to 100% for full green
      let lightness = 90; // Keeping lightness constant, but you can adjust this

      // For a more nuanced control over the color transition, you can also manipulate the hue.
      // Example: Start from a hue that represents a neutral color and move towards green (120) as activation increases
      let hue = this.activation * 120; // This will move from red (0) to green (120) based on activation

      return `background-color: hsl(${hue}, ${saturation}%, ${lightness}%);`;
    },
  },
};
</script>
