<template id="modal">
    <transition name="modal">
        <div class="modal-mask" @click="close" v-show="show">
            <div class="modal-container" @click.stop>
              <div class="slot-wrapper"><slot name="header"></slot></div>
              <div class="slot-wrapper"><slot name="body"></slot></div>
              <div class="slot-wrapper"><slot name="footer"></slot></div>
            </div>
        </div>
    </transition>
</template>

<script>
export default {
    props: ['show'],
    methods: {
    close: function () {
      this.$emit('close');
    }
  },
  mounted: function () {
    document.addEventListener("keydown", (e) => {
      if (this.show && e.keyCode == 27) {
        this.close();
      }
    });
  }
}
</script>


<style>
.modal-mask { 
    position: fixed;
    top: 0;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: rgba(0, 0, 0, 0.3);
    display: flex;
    justify-content: center;
    align-items: center;
}

.modal-container {
    padding: 1%;
    background-color: white;
    border-radius: 2px;
    box-shadow: 0 2px 8px black;
    overflow-x: auto;
    display: flex;
    flex-direction: column;
    max-width: 80%;
    max-height: 80%;
    min-width: 20%;
    min-height: 20%
}
.slot-wrapper{
  margin: 1% 1%;
}
.modal-enter-active, .modal-leave-active {
  transition: opacity .5s;
}
.modal-enter, .modal-leave-to {
  opacity: 0;
}
</style>
