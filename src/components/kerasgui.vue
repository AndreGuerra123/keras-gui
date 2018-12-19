<template>
  <transition appear>
    <div id="kerasgui">
      <navbar fixed-bottom class="kerastoolbar">
        <navbar-nav left>
          <btn-group>
            <btn type="info" class="navbar-btn" @click="import_json">Import</btn>
            <btn type="info" class="navbar-btn" @click="export_json">Export</btn>
            <btn type="info" class="navbar-btn" @click="edit_json">Edit</btn>
            <btn type="info" class="navbar-btn" @click="refresh_json">Refresh</btn>
            <btn type="primary" class="navbar-btn" @click="help_json">Help</btn>
          </btn-group>
        </navbar-nav>
        <navbar-nav right v-if="new_data">
          <btn-group>
            <btn type="danger" class="navbar-btn" @click="save_json">Save</btn>
            <btn type="danger" class="navbar-btn" @click="reset_json">Reset</btn>
          </btn-group>
        </navbar-nav>
      </navbar>

      <div id="kerasviewer" ref="kerasviewer"></div>
      <modal :show="importing" @close="close_importer">
        <h2 slot="header">Import File</h2>
        <input slot="body" type="file" class="form-control" @change="imported($event)">
        <div slot="footer">
          <btn type="primary" @click="close_importer">Close</btn>
        </div>
      </modal>
      <modal :show="exporting" @close="close_exporter">
        <h2 slot="header">Export File</h2>
        <input slot="body" type="text" v-model="filename">
        <div slot="footer">
          <btn type="default" @click="close_exporter">Close</btn>
          <btn type="primary" @click="exported">Export</btn>
        </div>
      </modal>
      <modal :show="editing" @close="close_editor">
        <h2 slot="header">Edit Source</h2>
        <div ref="keraseditor" slot="body">
        </div>
        <div slot="footer">
          <btn type="default" @click="close_editor">Close</btn>
          <btn type="primary" @click="edited">Save Edit</btn>
        </div>
      </modal>
      <modal :show="saving" @close="close_saver">
        <h2 slot="header">Save Source</h2>
        <div slot="body">
          <p>Do you really want to save the new model instance?</p>
          <p><strong>Note:</strong> This action will replace any existing model.</p>
        </div>
        <div slot="footer">
          <btn type="default" @click="close_editor">Close</btn>
          <btn type="primary" @click="saved">Save Model</btn>
        </div>
      </modal>
      <modal :show="resetting" @close="close_resetter">
        <h2 slot="header">Reset Source</h2>
        <div slot="body">
          <p>Do you really want to reset/delete the modified model instance?</p>
          <p><strong>Note:</strong> This action will delete permanently any modifications made to the model.</p>
        </div>
        <div slot="footer">
          <btn type="default" @click="close_resetter">Close</btn>
          <btn type="primary" @click="resetted">Reset Model</btn>
        </div>
      </modal>
      <modal :show="helping" @close="close_helper">
        <h2 slot="header">Help</h2>
        <div slot="footer">
          <btn type="primary" @click="close_helper">Close</btn>
        </div>
      </modal>
      <modal :show="node_editing" @close="close_node_editor">
        <h2 slot="header">Node Editor</h2>
        <div slot=body>
          <json-editor :schema="node_schema" :initial-value="node_data" @update-value="updatedNode($event)" theme="bootstrap3"
            icon="bootstrap3" noSelect2="true"></json-editor>
        </div>
        <div slot="footer">
          <btn type="default" @click="close_node_editor">Close</btn>
          <btn type="primary" @click="save_node_editor">Save Node</btn>
        </div>
      </modal>
    </div>
  </transition>
</template>

<script>
  let l = require('lodash')
  let dagre = require('dagre');
  let SVG = require('svg.js');
  let dat = require('dat.gui');
  require('svg.panzoom.js');
  let FileSaver = require('file-saver');
  import "vue-schema-based-json-editor";
  import modal from './modal.vue'
  import JSONEditor from 'jsoneditor'
  import 'jsoneditor/dist/jsoneditor.min.css'
  let compareVersions = require('compare-versions');

  import 'bootstrap/dist/css/bootstrap.css'
  import 'font-awesome/css/font-awesome.css'
  import {
    Navbar,
    NavbarNav,
    Btn,
    BtnGroup,
  } from 'uiv'
import { isArray } from 'util';

  export default {
    props: {
      json: {
        required: true,
        type: Object
      },
      rankdir: {
        default: "UD",
        type: String
      },
      nodesep: {
        default: 20,
        type: Number
      },
      edgesep: {
        default: 20,
        type: Number
      },
      ranksep: {
        default: 40,
        type: Number
      },
      marginx: {
        default: 0,
        type: Number
      },
      marginy: {
        default: 0,
        type: Number
      },
      nodewidth: {
        default: 200,
        type: Number
      },
      nodeheight: {
        default: 150,
        type: Number
      }
    },
    components: {
      modal,
      Navbar,
      NavbarNav,
      Btn,
      BtnGroup
    },
    data: function () {
      return {
        layer_ndx: "model",
        graph: {
          layer: []
        },
        merge_layers_ndx: [],
        unnamed_layer_count: 0,
        importing: false,
        exporting: false,
        editing: false,
        saving: false,
        resetting: false,
        helping: false,
        keraseditor: null,
        tmp_json: null,
        filename: null,
        node_editing: null,
        node_data: null,
        node_schema: null,
        new_node_data: null

      }
    },
    methods: {
      getVersion: function () {
        return l.get(this.json_data, 'keras_version', null)
      },
      clear: function () {
        this.view.clear();
        this.dgraf.g.setGraph({});
        this.dgraf.g.setDefaultEdgeLabel(function () {
          return {};
        });
        this.dgraf.svgd.clear()
        this.dgraf.marker = this.dgraf.svgd.marker(20, 10, function (add) {
          add.path("M20,0l10,5l-10,5l4,-5l-4,-5z").attr({
            "stroke-linecap": "round",
            "stroke-linejoin": "round"
          })
        })

      },
      isValidArray: function (arr) {
        return (!arr || l.isEmpty(arr)) ? false : true;
      },
      flatten_input: function (layer, arr) {
        var input_arr = [];
        if (!this.isValidArray(arr)) {
          if (this.layer_ndx == 0) return [];
          return [this.source_layers[this.layer_ndx - 1].config.name, layer];
        }
        arr.forEach((ar) => {
          var rec = this.flatten_input(layer, ar)
          if (rec == []) input_arr.push(rec)
        });
        return input_arr;
      },
      flattenArrayOfArrays: function (aa, r) {
        if (!r) r = []
        aa.forEach((a) => {
          if (a.constructor == Array) {
            this.flattenArrayOfArrays(a, r);
          } else {
            r.push(a);
          }
        });
        return r;
      },
      generateLayerName: function (idx) {
        return 'Untitled_Layer_' + idx;
      },
      getModelName: function () {
        return this.json_data.config.name || 'Untitled_Model'
      },
      addModel: function () {
        var name = this.getModelName()
        var class_name = this.json_data.class_name
        var version = this.getVersion() || "Unknown Keras Version"
        this.dgraf.nodes[name] = {
          label: name + "\n" + class_name + "\n" + version,
          color: this.get_info(this.json_data.class_name).color,
          width: this.nodewidth,
          height: this.nodeheight
        }
        if (this.json_data.config.input_layers) {
          this.flattenArrayOfArrays(this.json_data.config.input_layers).forEach((link) => {
            if (typeof link == "string") this.dgraf.edges.push([name, link])
          })
        } else if (this.isValidArray(this.json_data.config)) {
          this.dgraf.edges.push([name, this.json_data.config[0].config.name])
        }
        this.source_layers = this.json_data.config.layers ? this.json_data.config.layers : this.json_data.config
        this.source_layers.model = this.json_data
      },

      addLayers: function () {
        this.source_layers.forEach((layer, idx) => {
          if (layer.keras_version) return;
          this.layer_ndx = idx
          var name = layer.config.name || this.generateLayerName(idx)
          var class_name = layer.class_name
          this.dgraf.nodes[name] = {
            label: name + "\n" + class_name,
            color: this.get_info(class_name).color,
            width: this.nodewidth,
            height: this.nodeheight
          }
          if (class_name == "Merge") {
            this.merge_layers_ndx.push(idx)
          }
          if (this.layer_ndx != 0) {
            if (layer.inbound_nodes) {
              this.flattenArrayOfArrays(layer.inbound_nodes).forEach((edge) => {
                if (edge !== 0) this.dgraf.edges.push([edge, name])
              })
            } else if (this.layer_ndx != "model") {
              this.dgraf.edges.push([this.source_layers[this.layer_ndx - 1].config.name, name])
            }
          }
        })
        this.layer_ndx = 0
      },
      setGraph: function () {
        this.dgraf.g.setGraph({
          rankdir: this.rankdir,
          nodesep: this.nodesep,
          edgesep: this.edgesep,
          ranksep: this.ranksep,
          marginx: this.marginx,
          marginy: this.marginy,
          dims: this.dims
        })

        this.dgraf.edges.forEach((edge) => {
          this.dgraf.g.setEdge(edge[0], edge[1])
        })

        l.forOwn(this.dgraf.nodes, (nk, nv) => {
          this.dgraf.g.setNode(nv, nk)
        })

        dagre.layout(this.dgraf.g)
        var det = this.dgraf.g.graph()
        this.dgraf.svgd.size(det.width, det.height)
      },
      convertIdxLayer: function (i) {
        if (i === 0) {
          return "model"
        } else {
          return i - 1
        }
      },
      convertLayerIdx: function(l){
        if(l == "model"){
          return 0
        }else{
          return l+1
        }
      },
      draw_node: function (svgd, node, j) {

        let r = Math.min(node.width, node.height) / 2
        let idx = this.convertIdxLayer(j)
   
        var group = svgd.group().attr({
          "stroke-width": 0.5
        })
        group.rect(node.width, node.height).cx(node.x).cy(node.y).fill(node.color).attr({
          "fill-opacity": 0.2,
          "rx": r,
          "ry": r
        })

        var txt = group.text(node.label).x(node.x).y(node.y - 20).font({
          anchor: 'middle'
        })

        if (this.rankdir == "LR") txt.rotate(-90) ///initial

        //this adds the functionality of clicking in a node and a microeditor opens for edition.
        group.dblclick(() => {
          this.node_edit(idx)
        });

        //Add functionality
        if (idx != "model") {
          var add = group.group()
          add.text("+").attr("font-size", "40px")
          add.cx(node.x + node.width / 2).cy(node.y - (node.height / 2) + 20)
          if (this.rankdir == "LR") {
            add.rotate(-90)
          }
          add.click(() => {
            this.add_layer(j)
          });
        }

        //Remove funcionality
        if (idx != "model") {

        var addD = group.group()
        addD.text("-").attr("font-size", "40px")
        addD.cx(node.x + node.width / 2).cy(node.y - 15)
        if (this.rankdir == "LR") {
          addD.rotate(-90)
        }
        addD.click(()=>{
          this.remove_layer(idx)
        });
        }

        //Add below functionality

        var addL = group.group()
        addL.circle(25).attr({
          fill: "#fff",
          "stroke-width": 2
        })
        addL.path(
          "m5,10.6446 l3.57256,4.51989l3.57256,4.51989l3.57256,-4.51989l3.57256,-4.51989l-4.96673,0l0,-7.32094l-4.35678,0l0,7.32094l-4.96673,0z"
        )
        addL.cx(node.x + node.width / 2).cy(node.y + (node.height / 2) - 25)
        if (this.rankdir == "LR") {
          addL.rotate(-90)
        }
        addL.click(() => {
          this.add_layer_below(idx)
        })

        return group


      },
      get_above_layer: function(){
        return this.layer_ndx != 0 ? this.layer_ndx - 1 : "model"
      },
      
      add_layer_below: function (lndx) { //TODO: Identify model name (Note this approach in overall may be problematic since identifiying nodes by name (when name may be missing seems risky))
        this.layer_ndx = lndx;
        let current_name = lndx!="model" ? this.source_layers[lndx].config.name : "" 
        let current_length = this.tmp_json.config.length
        let new_layer = {
          "class_name": "Dense",
          "trainable": true,
          "config": {
            "b_constraint": null,
            "bias": true,
            "init": "uniform",
            "output_dim": null,
            "input_dim": null,
            "W_regularizer": null,
            "activity_regularizer": null,
            "W_constraint": null,
            "trainable": true,
            "name": "dense_" + (current_length),
            "b_regularizer": null,
            "activation": "tanh"
          },
          "name": "dense_" + (current_length) ,
          "inbound_nodes": [
            [
              [
                current_name,
                0,
                0
              ]
            ]
          ]
        
        }

        this.tmp_json = Object.create(this.tmp_json)
        this.tmp_json.config.push(new_layer)
        this.render()
      },
      add_layer: function (lndx) { //TODO: Identify model name (Note this approach in overall may be problematic since identifiying nodes by name (when name may be missing seems risky))
        this.layer_ndx = lndx;
        let above_name = (lndx != "model") ? this.source_layers[lndx-2].config.name : this.tmp_json.config.model.name
        let current_length = this.tmp_json.config.length
        let new_layer = {
          "class_name": "Dense",
          "trainable": true,
          "config": {
            "b_constraint": null,
            "bias": true,
            "init": "uniform",
            "output_dim": null,
            "input_dim": null,
            "W_regularizer": null,
            "activity_regularizer": null,
            "W_constraint": null,
            "trainable": true,
            "name": "dense_" + (current_length),
            "b_regularizer": null,
            "activation": "tanh"
          },
          "name": "dense_" + (current_length) ,
          "inbound_nodes": [
            [
              [
                above_name,
                0,
                0
              ]
            ]
          ]
        
        }

        this.tmp_json = Object.create(this.tmp_json)
        this.tmp_json.config.push(new_layer)
        this.render()
      },

      remove_layer: function (lndx) {//TODO: How to implement?
        this.tmp_json = Object.create(this.tmp_json)
        
        this.render()
      },

      draw_edge: function (svgd, points, marker) {
        var str = "M " + points[0].x + ' ' + points[0].y + 'C '
        var len = points.length
        if (len == 2)
          str += points[0].x + ' ' + points[0].y + ' ' + points[1].x + ' ' + points[1].y + ' '
        else if (len == 3)
          str += points[1].x + ' ' + points[1].y + ' ' + points[1].x + ' ' + points[1].y + ' '
        else if (len == 4)
          str += points[1].x + ' ' + points[1].y + ' ' + points[2].x + ' ' + points[2].y + ' '
        else
          str += points[1].x + ' ' + points[1].y + ' ' + points[len - 2].x + ' ' + points[len - 2].y + ' '

        str += points[len - 1].x + ' ' + points[len - 1].y

        var edge = svgd.path(str).fill('none').stroke({
          width: 1
        }).attr({
          "stroke-linecap": "round",
          "stroke-linejoin": "round"
        })
        edge.marker('end', marker)
        return edge;
      },
      drawGraph: function () {
        this.dgraf.g.edges().forEach((e) => {
          this.draw_edge(this.dgraf.svgd, this.dgraf.g.edge(e).points, this.dgraf.marker)
        })

        this.dgraf.g.nodes().forEach((node, nidx) => {
          if (nidx == 0) { //TODO:
            this.graph.layer.model = this.draw_node(this.dgraf.svgd, this.dgraf.nodes[node], nidx)
          } else {
            this.graph.layer[nidx - 1] = this.draw_node(this.dgraf.svgd, this.dgraf.nodes[node], nidx)
          }
        })
      },
      boxGraph: function () {
        var d = this.dgraf.svgd.bbox()
        var div = {
          w: this.$refs.kerasviewer.clientWidth,
          h: this.$refs.kerasviewer.clientHeight
        }
        var r = 1
        if (d.w / d.h > div.w / div.h) {
          r = div.w / d.w
        } else {
          r = div.h / d.h
        }
        this.dgraf.svgd.attr({
          transform: "matrix(" + r + ",0,0," + r + ",0,0)"
        })
      },
      buildGraphModel: function () {
        this.clear()
        this.addModel()
        this.addLayers()
      },
      showGraph: function () {
        this.setGraph()
        this.drawGraph()
        this.boxGraph()
      },
      render: function () {
        this.buildGraphModel();
        this.showGraph();
      },

      ////////NODE EDITOR//////////

      get_info: function (classs) {
        let layerinfo = l.get(this.config.info, classs, null) || {
          help: "No description provided.",
          color: "#ffffff",
          args: []
        }
        layerinfo.meta = l.get(this.config.meta, classs, null) || "Unknown meta-class."
        return layerinfo
      },

      is_model_node: function (classs) {
        return l.some(this.config.enums.Models, (x) => x === classs);

      },

      is_layer_node: function (classs) {
        return l.some(this.config.enums.Layers, (x) => x === classs);

      },

      is_optimizer_node: function (classs) {
        return l.some(this.config.enums.Optimizers, (x) => x === classs);
      },

      get_arg_choices: function (arg) {
        return l.get(this.config.enums, arg, null)
      },

      get_arg_description: function (arg) {
        return l.get(this.config.help, arg, null)
      },
      get_core_config_props: function (classs) {
        if (this.is_model_node(classs)) {
          return this.config.model_config_core_schema;
        } else if (this.is_layer_node(classs)) {
          return this.config.layer_config_core_schema;
        } else if (this.is_optimizer_node(classs)) {
          return this.config.optimizer_config_core_schema;
        } else {
          return {}
        }
      },
      get_default_config_props: function (classs) {
        let args_props = {};
        let aa = l.get(this.get_info(classs), "args", [])
        l.map(aa, (a) => {
          let arg_props = l.get(this.config.default_schemas, a, {})
          this.get_arg_choices(a) ? l.set(arg_props, 'enum', this.get_arg_choices(a)) : ""
          this.get_arg_description(a) ? l.set(arg_props, 'description', l.upperFirst(this.get_arg_description(a))) :
            ""
          args_props[a] = arg_props
        })
        return args_props;
      },

      get_override_config_props: function (classs) {
        return l.get(this.config.override_schema, classs, {})
      },

      get_config_props: function (classs) {
        return {
          "config": {
            "type": "object",
            "title": "Configuration",
            "description": "Edit this node configuration.",
            "properties": l.merge(this.get_core_config_props(classs), this.get_default_config_props(classs), this.get_override_config_props(
              classs))
          }
        }
      },
      get_class_name_props: function () {
        return {
          "class_name": {
            "type": "string",
            "title": "Class",
            "enum": l.union(this.config.enums.Models, this.config.enums.Layers, this.config.enums.Optimizers)
          }
        }
      },
      get_meta_class_props: function (classs) {
        if (this.is_model_node(classs)) {
          return this.config.model_meta_core_schema;
        } else if (this.is_layer_node(classs)) {
          return this.config.layer_meta_core_schema;
        } else if (this.is_optimizer_node(classs)) {
          return this.config.optimizer_meta_core_schema;
        } else {
          return {}
        }
      },
      get_node_schema: function (node_data) {
        var classs = node_data.class_name
        var info = this.get_info(classs)
        var name = node_data.config.name

        return {
          "type": "object",
          "title": l.upperFirst(l.join([this.layer_ndx, name, classs, info.meta], " :: ")),
          "description": l.upperFirst(info.help),
          "properties": l.merge(this.get_class_name_props(), this.get_meta_class_props(classs), this.get_config_props(
            classs)),
          "required": ["class_name", "config"]

        }
      },
      node_edit: function (lidx) {
        this.layer_ndx = lidx
        this.node_data = this.source_layers[this.layer_ndx] //initialize node_data
        this.node_schema = this.get_node_schema(this.node_data), //initialize node_schema
          this.node_editing = true;
      },
      updatedNode: function (e) {
        this.new_node_data = e.value
      },
      save_node_editor: function () {
        var classs = l.get(this.new_node_data, "class_name", null)
        this.tmp_json = Object.create(this.tmp_json)
        if (this.is_model_node(classs)) {
          this.tmp_json = l.merge(this.tmp_json, this.new_node_data)
        }
        this.tmp_json.config[this.layer_ndx] = this.new_node_data;


        this.close_node_editor()
      },
      close_node_editor: function () {
        this.node_editing = false;
        this.render()

      },
      /////////IMPORTER//////////
      import_json: function () {
        this.importing = true;
      },
      close_importer: function () {
        this.importing = false;
      },
      imported: function (event) {
        const reader = new FileReader();
        let result;
        reader.onloadend = (e) => {
          result = JSON.parse(e.target.result)
        }
        reader.readAsText(event.target.files[0])
        this.tmp_json = result
        this.close_importer()
        this.render()
      },
      //////////EXPORTER/////////
      export_json: function () {
        this.clearpops()
        this.filename = this.getModelName() + '_' + new Date().getTime() + ".json"
        this.exporting = true;
      },
      close_exporter: function () {
        this.exporting = false;
      },
      exported: function () {
        var file = new File([JSON.stringify(this.json_data)], this.filename, {
          type: "text/plain;charset=utf-8"
        });
        FileSaver.saveAs(file);
        this.close_exporter()
      },
      ////////////EDITOR/////////
      edit_json: function () {
        this.clearpops()
        this.keraseditor.set(this.json_data);
        this.editing = true;
      },
      close_editor: function () {
        this.editing = false;
      },
      edited: function () {
        this.tmp_json = this.keraseditor.get()
        this.close_editor()
        this.render()
      },
      ////////////SAVER//////////
      save_json: function () {
        this.clearpops()
        this.saving = true;
      },
      close_saver: function () {
        this.saving = false;
      },
      saved: function () {
        this.json = this.json_data;
        this.close_saver();
        this.render();
      },
      /////////////RESETER/////////

      reset_json: function () {
        this.clearpops()

        this.resetting = true;
      },
      close_resetter: function () {
        this.resetting = false;

      },
      resetted: function () {
        this.tmp_json = this.json
        this.close_resetter();
        this.render();
      },
      /////////////REFRESH/////////
      refresh_json: function () {
        this.clearpops()

        this.render()
      },
      ///////////HELP//////////////
      help_json: function () {
        this.clearpops()
        this.helping = true;
      },
      close_helper: function () {
        this.helping = false;
      },
      clearpops: function () {
        this.importing = false;
        this.exporting = false;
        this.editing = false;
        this.saving = false;
        this.resetting = false;
        this.helping = false;
      }

    },
    computed: {
      config: function () {
        //TODO: change first output to the new keras 2.0 config file
        if (this.getVersion()) {
          return (compareVersions(this.getVersion(), "2.0.0") >= 0) ? require('../assets/keras_config.js') : require(
            '../assets/keras_config.js');

        } else { //LOAD THE MOST RECENT
          return require('../assets/keras_config.js');
        }
      },
      json_data: function () {
        return this.tmp_json ? this.tmp_json : this.json
      },
      new_data: function () {
        return (this.tmp_json && this.tmp_json !== this.json) ? true : false;
      },
      draw: function () {
        return SVG(this.$refs.kerasviewer).panZoom()
      },
      view: function () {
        return this.draw.group().attr({
          id: "kerasviewer"
        })
      },
      dgraf: function () {
        return {
          svgd: this.view,
          nodes: {},
          edges: [],
          g: new dagre.graphlib.Graph()

        }
      },
      dims: function () {
        if ((this.rankdir == "LR" && this.nodewidth > this.nodeheight) || (this.rankdir == "UD" && this.nodewidth <
            this.nodeheight)) {
          return {
            w: this.nodeheight,
            h: this.nodewidth
          }
        } else {
          return {
            h: this.nodeheight,
            w: this.nodewidth
          }
        }
      }
    },
    mounted() {
      this.tmp_json = this.json
      this.keraseditor = new JSONEditor(this.$refs.keraseditor, {})
      this.render()
    },

  }
</script>

<style>
  #kerasgui {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column
  }

  #kerasviewer {
    flex: 1 1 0;
  }

  .kerastoolbar {
    width: 30%;
    margin: 1% auto;
    background-color: transparent;
    border-color: transparent;
  }

  #node-editor {
    max-width: 50%;
    max-height: 50%;
  }
</style>