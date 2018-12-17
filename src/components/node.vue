<template>
 <svg>
  <rect height="150px" width="150px" y="50px"></rect>
  <text id="textwrap" y="50px" font-size="12">{{text}}</text>
 </svg>
</template>

<script>
    let d3 = require('d3');
    let d3plus = require('d3plus');
    
    export default {
        props:{
            text:{
                default: "",
                type: String
            }
        },
        mounted: function(){
            d3plus.textwrap().container(d3.select("#textwrap")).resize(true).draw();
        }
    }
</script>



