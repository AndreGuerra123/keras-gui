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
        <navbar-nav right>
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
          <json-editor :schema="node_schema" :initial-value="node_initial" @update-value="saved_node($event)" theme="bootstrap3"
            icon="fontawesome4"></json-editor>
        </div>
        <div slot="footer">
          <btn type="default" @click="close_node_editor">Close</btn>
          <btn type="primary" @click="saved_node">Save Node</btn>
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
  let KerasModelDefaults = require('../assets/keras_model_defaults_v2.js');
  let KerasModelConfigs = require('../assets/keras_model_configs_v2.js');
  let FileSaver = require('file-saver');
  import "vue-schema-based-json-editor";
  import modal from './modal.vue'
  import JSONEditor from 'jsoneditor'
  import 'jsoneditor/dist/jsoneditor.min.css'

  import 'bootstrap/dist/css/bootstrap.css'
  import {
    Navbar,
    NavbarNav,
    Btn,
    BtnGroup,
  } from 'uiv'

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
        keras_args: KerasModelDefaults.keras_args,
        keras_conf: KerasModelConfigs.keras_conf,
        keras_choices: KerasModelConfigs.keras_choices,
        keras_help: KerasModelConfigs.keras_help,
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
        node_schema: {
          type: Object
        },
        node_initial: {}

      }
    },
    methods: {

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
        var version = this.json_data.keras_version || "Unknown"
        this.dgraf.nodes[name] = {
          label: name + "\n" + class_name + "\n" + version,
          color: this.keras_args[this.json_data.class_name].color,
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
            color: this.keras_args[class_name].color,
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

      ////////NODE EDITOR//////////
      get_node_initial: function () {
        return {
          stringExample: 'a default string example',
          booleanExample: true,
          numberExample: 123.4,
          integerExample: 124,
          nullExample: null,
          objectExample: {
            propertyExample1: '',
            propertyExample2: 0,
          },
          arrayExample: [
            'default item 1',
            'default item 2',
          ],
          readOnlyExample: 'abc',
          enumExample: 'enum 1',
          colorExample: '#000000',
          textareaExample: '',
          patternExample: 'abc',
          imagePreviewExample: 'http://image2.sina.com.cn/bj/art/2004-08-02/U91P52T4D51657F160DT20040802125523.jpg',
          markdownExample: '###### markdown title and code example\n\n```js\nfunction foo(bar) {\n    console.log(bar);\n}\n```',
          codeExample: 'function foo(bar) {\n    console.log(bar);\n}\n',
          itemTitleExample: [{
              propertyExample1: 'foo',
              propertyExample2: 1,
            },
            {
              propertyExample1: 'bar',
              propertyExample2: 2,
            },
            {
              propertyExample1: 'baz',
              propertyExample2: 3,
            },
            {
              propertyExample1: 'abc',
              propertyExample2: 4,
            },
            {
              propertyExample1: 'def',
              propertyExample2: 5,
            },
            {
              propertyExample1: 'ghi',
              propertyExample2: 6,
            },
          ],
          optionalObjectExample: {},
          collapsedObjectExample: {},
          emptyEnumExample: '',
          uploadFileExample: '',
          requiredWhenExample: {
            kind: 'enum 1',
            propertyExample1: 0,
            propertyExample3: 0,
            propertyExample4: '',
          },
          checkboxBooleanExample: true,
          enumTitlesExample: 'enum 1',
          itemTitleEnumTitleExample: [{
            propertyExample1: 'bar',
          }, ],
          selectBooleanExample: false,
          select2BooleanExample: false,
          stringEnumSelectExample: 'enum 1',
          stringEnumRadioboxExample: 'enum 1',
          numberEnumSelectExample: 1,
          numberEnumRadioboxExample: 1,
          classNameExample: '',
          enumArrayExample: [],
          enumArraySelect2Example: [],
        }
      },
      get_node_schema: function (classs) {
        return {
          "type": "object",
          "title": "GUI:",
          "description": "a description example",
          "properties": {
            "stringExample": {
              "type": "string",
              "title": "A string example",
              "description": "a string description example",
              "default": "a default string example",
              "minLength": 15,
              "maxLength": 20,
              "propertyOrder": 1
            },
            "booleanExample": {
              "type": "boolean",
              "title": "A boolean example",
              "description": "a boolean description example",
              "default": true,
              "propertyOrder": 2
            },
            "numberExample": {
              "type": "number",
              "title": "A number example",
              "description": "a number description example",
              "default": 123.4,
              "minimum": 10,
              "exclusiveMinimum": true,
              "maximum": 1000,
              "exclusiveMaximum": true,
              "propertyOrder": 3
            },
            "integerExample": {
              "type": "integer",
              "title": "A integer example",
              "description": "a integer description example",
              "default": 124,
              "multipleOf": 2,
              "propertyOrder": 4
            },
            "nullExample": {
              "type": "null",
              "title": "A null example",
              "description": "a null description example",
              "default": null,
              "propertyOrder": 5
            },
            "objectExample": {
              "type": "object",
              "title": "A object example",
              "description": "a object description example",
              "properties": {
                "propertyExample1": {
                  "type": "string"
                },
                "propertyExample2": {
                  "type": "number"
                }
              },
              "default": {},
              "required": [
                "propertyExample1",
                "propertyExample2"
              ],
              "propertyOrder": 6
            },
            "arrayExample": {
              "type": "array",
              "title": "A array example",
              "description": "a array description example",
              "items": {
                "type": "string",
                "maxLength": 15
              },
              "default": [
                "default item 1",
                "default item 2"
              ],
              "minItems": 1,
              "uniqueItems": true,
              "propertyOrder": 7
            },
            "readOnlyExample": {
              "type": "string",
              "readonly": true,
              "default": "abc",
              "propertyOrder": 8
            },
            "readOnlyAndOptionalExample": {
              "type": "string",
              "readonly": true,
              "default": "abc",
              "propertyOrder": 9
            },
            "enumExample": {
              "type": "string",
              "enum": [
                "enum 1",
                "enum 2"
              ],
              "propertyOrder": 10
            },
            "optionalExample": {
              "type": "string",
              "propertyOrder": 11
            },
            "optionalAndDefaultExample": {
              "type": "string",
              "default": "abc",
              "propertyOrder": 12
            },
            "booleanOptionalExample": {
              "type": "boolean",
              "propertyOrder": 13
            },
            "colorExample": {
              "type": "string",
              "format": "color",
              "default": "#000000",
              "propertyOrder": 14
            },
            "textareaExample": {
              "type": "string",
              "format": "textarea",
              "propertyOrder": 15
            },
            "patternExample": {
              "type": "string",
              "pattern": "^[A-z]{3}$",
              "default": "abc",
              "propertyOrder": 16
            },
            "imagePreviewExample": {
              "type": "string",
              "default": "http://image2.sina.com.cn/bj/art/2004-08-02/U91P52T4D51657F160DT20040802125523.jpg",
              "propertyOrder": 17
            },
            "markdownExample": {
              "type": "string",
              "format": "markdown",
              "default": "###### markdown title and code example\n\n```js\nfunction foo(bar) {\n    console.log(bar);\n}\n```",
              "propertyOrder": 18
            },
            "codeExample": {
              "type": "string",
              "format": "code",
              "default": "function foo(bar) {\n    console.log(bar);\n}\n",
              "propertyOrder": 19
            },
            "itemTitleExample": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "propertyExample1": {
                    "type": "string"
                  },
                  "propertyExample2": {
                    "type": "number"
                  }
                },
                "required": [
                  "propertyExample1",
                  "propertyExample2"
                ],
                "collapsed": true
              },
              "default": [{
                  "propertyExample1": "foo",
                  "propertyExample2": 1
                },
                {
                  "propertyExample1": "bar",
                  "propertyExample2": 2
                },
                {
                  "propertyExample1": "baz",
                  "propertyExample2": 3
                },
                {
                  "propertyExample1": "abc",
                  "propertyExample2": 4
                },
                {
                  "propertyExample1": "def",
                  "propertyExample2": 5
                },
                {
                  "propertyExample1": "ghi",
                  "propertyExample2": 6
                }
              ],
              "propertyOrder": 20
            },
            "optionalObjectExample": {
              "type": "object",
              "properties": {
                "propertyExample1": {
                  "type": "string"
                },
                "propertyExample2": {
                  "type": "number"
                }
              },
              "maxProperties": 1,
              "minProperties": 0,
              "propertyOrder": 21
            },
            "collapsedObjectExample": {
              "type": "object",
              "properties": {
                "propertyExample1": {
                  "type": "string"
                }
              },
              "collapsed": true,
              "propertyOrder": 22
            },
            "emptyEnumExample": {
              "type": "string",
              "enum": [],
              "propertyOrder": 23
            },
            "uploadFileExample": {
              "type": "string",
              "format": "base64",
              "propertyOrder": 24
            },
            "requiredWhenExample": {
              "type": "object",
              "properties": {
                "kind": {
                  "type": "string",
                  "enum": [
                    "enum 1",
                    "enum 2",
                    "enum 3"
                  ]
                },
                "propertyExample1": {
                  "type": "number",
                  "requiredWhen": [
                    "kind",
                    "===",
                    "enum 1"
                  ]
                },
                "propertyExample2": {
                  "type": "string",
                  "requiredWhen": [
                    "kind",
                    "===",
                    "enum 2"
                  ]
                },
                "propertyExample3": {
                  "type": "number",
                  "requiredWhen": [
                    "kind",
                    "in",
                    [
                      "enum 1",
                      "enum 2"
                    ]
                  ]
                },
                "propertyExample4": {
                  "type": "string"
                },
                "propertyExample5": {
                  "type": "string"
                },
                "propertyExample6": {
                  "type": "string",
                  "optionalWhen": [
                    "kind",
                    "===",
                    "enum 2"
                  ]
                }
              },
              "required": [
                "kind",
                "propertyExample4"
              ],
              "propertyOrder": 25
            },
            "checkboxBooleanExample": {
              "type": "boolean",
              "default": true,
              "format": "checkbox",
              "propertyOrder": 26
            },
            "enumTitlesExample": {
              "type": "string",
              "enum": [
                "enum 1",
                "enum 2"
              ],
              "enumTitles": [
                "enum title 1",
                "enum title 2"
              ],
              "propertyOrder": 27
            },
            "itemTitleEnumTitleExample": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "propertyExample1": {
                    "type": "string",
                    "enum": [
                      "foo",
                      "bar"
                    ],
                    "enumTitles": [
                      "foo title",
                      "bar title"
                    ]
                  }
                },
                "required": [
                  "propertyExample1"
                ]
              },
              "default": [{
                "propertyExample1": "bar"
              }],
              "propertyOrder": 28
            },
            "selectBooleanExample": {
              "type": "boolean",
              "default": false,
              "format": "select",
              "propertyOrder": 28
            },
            "select2BooleanExample": {
              "type": "boolean",
              "default": false,
              "format": "select2",
              "propertyOrder": 30
            },
            "stringEnumSelectExample": {
              "type": "string",
              "format": "select",
              "enum": [
                "enum 1",
                "enum 2"
              ],
              "enumTitles": [
                "enum title 1",
                "enum title 2"
              ],
              "propertyOrder": 31
            },
            "stringEnumRadioboxExample": {
              "type": "string",
              "format": "radiobox",
              "enum": [
                "enum 1",
                "enum 2"
              ],
              "enumTitles": [
                "enum title 1",
                "enum title 2"
              ],
              "propertyOrder": 32
            },
            "numberEnumSelectExample": {
              "type": "number",
              "format": "select",
              "enum": [
                1,
                2
              ],
              "enumTitles": [
                "one",
                "two"
              ],
              "propertyOrder": 33
            },
            "numberEnumRadioboxExample": {
              "type": "number",
              "format": "radiobox",
              "enum": [
                1,
                2
              ],
              "enumTitles": [
                "one",
                "two"
              ],
              "propertyOrder": 34
            },
            "classNameExample": {
              "type": "string",
              "className": "custom-class-string",
              "propertyOrder": 35
            },
            "enumArrayExample": {
              "type": "array",
              "items": {
                "type": "string"
              },
              "enum": [
                "foo",
                "bar"
              ],
              "enumTitle": [
                "foo title",
                "bar title"
              ],
              "propertyOrder": 36
            },
            "enumArraySelect2Example": {
              "type": "array",
              "items": {
                "type": "string"
              },
              "enum": [
                "foo",
                "bar"
              ],
              "enumTitle": [
                "foo title",
                "bar title"
              ],
              "format": "select2",
              "propertyOrder": 37
            }
          },
          "required": [
            "stringExample",
            "booleanExample",
            "numberExample",
            "integerExample",
            "nullExample",
            "objectExample",
            "arrayExample",
            "readOnlyExample",
            "enumExample",
            "colorExample",
            "textareaExample",
            "patternExample",
            "imagePreviewExample",
            "markdownExample",
            "codeExample",
            "performanceExample",
            "itemTitleExample",
            "optionalObjectExample",
            "propertyOrderExample",
            "collapsedObjectExample",
            "emptyEnumExample",
            "uploadFileExample",
            "requiredWhenExample",
            "checkboxBooleanExample",
            "enumTitlesExample",
            "itemTitleEnumTitleExample",
            "selectBooleanExample",
            "select2BooleanExample",
            "stringEnumSelectExample",
            "stringEnumRadioboxExample",
            "numberEnumSelectExample",
            "numberEnumRadioboxExample",
            "classNameExample",
            "enumArrayExample",
            "enumArraySelect2Example"
          ]
        }
      },
      node_edit: function (lidx) {
        if (lidx != "model") {

          var node_class = this.source_layers[lidx].class_name

          this.node_initial = this.get_node_initial() //data

          this.node_schema = this.get_node_schema(node_class) //schema

          this.node_editing = true;

        }
      },
      saved_node: function () {

        this.close_node_editing()

        //TODO:  new data ?
        //this.tmp_data = this.extractAll()

        this.render()

      },
      close_node_editor: function () {
        this.node_schema = {
          type: Object
        }
        this.node_initial = {};
        this.node_editing = false;
      },
      draw_node: function (svgd, node, j) {

        var r;
        if (j === 0) {
          j = "model"
          r = Math.min(node.width, node.height) / 2
        } else {
          j--
        }
        var group = svgd.group().attr({
          "data-index": j,
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
        group.click(() => {
          this.node_edit(j)
        });

        return group


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
        this.tmp_data = null;
        this.json_data = this.json;
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
    margin: 5% auto;
    background-color: transparent;
    border-color: transparent;
  }
</style>