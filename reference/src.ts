/** @license See the LICENSE file. */

// This code is auto-generated, do not modify this file!
const version = '0.0.0';
export {version};

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import './flags';

export {IAttrValue, INameAttrList, INodeDef, ITensor, ITensorShape} from './data/compiled_api';
export {GraphModel, loadGraphModel, loadGraphModelSync} from './executor/graph_model';
export {deregisterOp, registerOp} from './operations/custom_op/register';
export {GraphNode, OpExecutor} from './operations/types';
export {version as version_converter} from './version';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {DataType, env} from '@tensorflow/tfjs-core';

import * as tensorflow from '../data/compiled_api';

import {getRegisteredOp} from './custom_op/register';
import {getNodeNameAndIndex} from './executors/utils';
import * as arithmetic from './op_list/arithmetic';
import * as basicMath from './op_list/basic_math';
import * as control from './op_list/control';
import * as convolution from './op_list/convolution';
import * as creation from './op_list/creation';
import * as dynamic from './op_list/dynamic';
import * as evaluation from './op_list/evaluation';
import * as graph from './op_list/graph';
import * as hashTable from './op_list/hash_table';
import * as image from './op_list/image';
import * as logical from './op_list/logical';
import * as matrices from './op_list/matrices';
import * as normalization from './op_list/normalization';
import * as reduction from './op_list/reduction';
import * as sliceJoin from './op_list/slice_join';
import * as sparse from './op_list/sparse';
import * as spectral from './op_list/spectral';
import * as string from './op_list/string';
import * as transformation from './op_list/transformation';
import {Graph, InputParamValue, Node, OpMapper, ParamValue} from './types';

export class OperationMapper {
  private static _instance: OperationMapper;

  private opMappers: {[key: string]: OpMapper};

  // Singleton instance for the mapper
  public static get Instance() {
    return this._instance || (this._instance = new this());
  }

  // Loads the op mapping from the JSON file.
  private constructor() {
    const ops = [
      arithmetic, basicMath, control, convolution, creation, dynamic,
      evaluation, graph, hashTable, image, logical, matrices, normalization,
      reduction, sliceJoin, sparse, spectral, string, transformation
    ];
    const mappersJson: OpMapper[] = [].concat(...ops.map(op => op.json));

    this.opMappers = mappersJson.reduce<{[key: string]: OpMapper}>(
        (map, mapper: OpMapper) => {
          map[mapper.tfOpName] = mapper;
          return map;
        },
        {});
  }

  // Converts the model inference graph from Tensorflow GraphDef to local
  // representation for TensorFlow.js API
  transformGraph(
      graph: tensorflow.IGraphDef,
      signature: tensorflow.ISignatureDef = {}): Graph {
    const tfNodes = graph.node;
    const placeholders: Node[] = [];
    const weights: Node[] = [];
    const initNodes: Node[] = [];
    const nodes = tfNodes.reduce<{[key: string]: Node}>((map, node) => {
      map[node.name] = this.mapNode(node);
      if (node.op.startsWith('Placeholder')) {
        placeholders.push(map[node.name]);
      } else if (node.op === 'Const') {
        weights.push(map[node.name]);
      } else if (node.input == null || node.input.length === 0) {
        initNodes.push(map[node.name]);
      }
      return map;
    }, {});

    let inputs: Node[] = [];
    const outputs: Node[] = [];
    let inputNodeNameToKey: {[key: string]: string} = {};
    let outputNodeNameToKey: {[key: string]: string} = {};
    if (signature != null) {
      inputNodeNameToKey = this.mapSignatureEntries(signature.inputs);
      outputNodeNameToKey = this.mapSignatureEntries(signature.outputs);
    }
    const allNodes = Object.keys(nodes);
    allNodes.forEach(key => {
      const node = nodes[key];
      node.inputNames.forEach((name, index) => {
        const [nodeName, , outputName] = getNodeNameAndIndex(name);
        const inputNode = nodes[nodeName];
        if (inputNode.outputs != null) {
          const outputIndex = inputNode.outputs.indexOf(outputName);
          if (outputIndex !== -1) {
            const inputName = `${nodeName}:${outputIndex}`;
            // update the input name to use the mapped output index directly.
            node.inputNames[index] = inputName;
          }
        }
        node.inputs.push(inputNode);
        inputNode.children.push(node);
      });
    });

    // if signature has not outputs set, add any node that does not have
    // outputs.
    if (Object.keys(outputNodeNameToKey).length === 0) {
      allNodes.forEach(key => {
        const node = nodes[key];
        if (node.children.length === 0) {
          outputs.push(node);
        }
      });
    } else {
      Object.keys(outputNodeNameToKey).forEach(name => {
        const [nodeName, ] = getNodeNameAndIndex(name);
        const node = nodes[nodeName];
        if (node != null) {
          node.signatureKey = outputNodeNameToKey[name];
          outputs.push(node);
        }
      });
    }

    if (Object.keys(inputNodeNameToKey).length > 0) {
      Object.keys(inputNodeNameToKey).forEach(name => {
        const [nodeName, ] = getNodeNameAndIndex(name);
        const node = nodes[nodeName];
        if (node) {
          node.signatureKey = inputNodeNameToKey[name];
          inputs.push(node);
        }
      });
    } else {
      inputs = placeholders;
    }

    let functions = {};
    if (graph.library != null && graph.library.function != null) {
      functions = graph.library.function.reduce((functions, func) => {
        functions[func.signature.name] = this.mapFunction(func);
        return functions;
      }, {} as {[key: string]: Graph});
    }

    const result: Graph =
        {nodes, inputs, outputs, weights, placeholders, signature, functions};

    if (initNodes.length > 0) {
      result.initNodes = initNodes;
    }

    return result;
  }

  private mapSignatureEntries(entries: {[k: string]: tensorflow.ITensorInfo}) {
    return Object.keys(entries || {})
        .reduce<{[key: string]: string}>((prev, curr) => {
          prev[entries[curr].name] = curr;
          return prev;
        }, {});
  }

  private mapNode(node: tensorflow.INodeDef): Node {
    // Unsupported ops will cause an error at run-time (not parse time), since
    // they may not be used by the actual execution subgraph.
    const mapper =
        getRegisteredOp(node.op) || this.opMappers[node.op] || {} as OpMapper;
    if (node.attr == null) {
      node.attr = {};
    }

    const newNode: Node = {
      name: node.name,
      op: node.op,
      category: mapper.category,
      inputNames:
          (node.input ||
           []).map(input => input.startsWith('^') ? input.slice(1) : input),
      inputs: [],
      children: [],
      inputParams: {},
      attrParams: {},
      rawAttrs: node.attr,
      outputs: mapper.outputs
    };

    if (mapper.inputs != null) {
      newNode.inputParams =
          mapper.inputs.reduce<{[key: string]: InputParamValue}>(
              (map, param) => {
                map[param.name] = {
                  type: param.type,
                  inputIndexStart: param.start,
                  inputIndexEnd: param.end
                };
                return map;
              },
              {});
    }
    if (mapper.attrs != null) {
      newNode.attrParams =
          mapper.attrs.reduce<{[key: string]: ParamValue}>((map, param) => {
            const type = param.type;
            let value = undefined;
            switch (param.type) {
              case 'string':
                value = getStringParam(
                    node.attr, param.tfName, param.defaultValue as string);

                if (value === undefined && !!param.tfDeprecatedName) {
                  value = getStringParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as string);
                }
                break;
              case 'string[]':
                value = getStringArrayParam(
                    node.attr, param.tfName, param.defaultValue as string[]);

                if (value === undefined && !!param.tfDeprecatedName) {
                  value = getStringArrayParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as string[]);
                }
                break;
              case 'number':
                value = getNumberParam(
                    node.attr, param.tfName,
                    (param.defaultValue || 0) as number);
                if (value === undefined && !!param.tfDeprecatedName) {
                  value = getNumberParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as number);
                }
                break;
              case 'number[]':
                value = getNumericArrayParam(
                    node.attr, param.tfName, param.defaultValue as number[]);
                if (value === undefined && !!param.tfDeprecatedName) {
                  value = getNumericArrayParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as number[]);
                }
                break;
              case 'bool':
                value = getBoolParam(
                    node.attr, param.tfName, param.defaultValue as boolean);
                if (value === undefined && !!param.tfDeprecatedName) {
                  value = getBoolParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as boolean);
                }
                break;
              case 'bool[]':
                value = getBoolArrayParam(
                    node.attr, param.tfName, param.defaultValue as boolean[]);
                if (value === undefined && !!param.tfDeprecatedName) {
                  value = getBoolArrayParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as boolean[]);
                }
                break;
              case 'shape':
                value = getTensorShapeParam(
                    node.attr, param.tfName, param.defaultValue as number[]);
                if (value === undefined && !!param.tfDeprecatedName) {
                  value = getTensorShapeParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as number[]);
                }
                break;
              case 'shape[]':
                value = getTensorShapeArrayParam(
                    node.attr, param.tfName, param.defaultValue as number[][]);
                if (value === undefined && !!param.tfDeprecatedName) {
                  value = getTensorShapeArrayParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as number[][]);
                }
                break;
              case 'dtype':
                value = getDtypeParam(
                    node.attr, param.tfName, param.defaultValue as DataType);
                if (value === undefined && !!param.tfDeprecatedName) {
                  value = getDtypeParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as DataType);
                }
                break;
              case 'dtype[]':
                value = getDtypeArrayParam(
                    node.attr, param.tfName, param.defaultValue as DataType[]);
                if (value === undefined && !!param.tfDeprecatedName) {
                  value = getDtypeArrayParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as DataType[]);
                }
                break;
              case 'func':
                value = getFuncParam(
                    node.attr, param.tfName, param.defaultValue as string);
                if (value === undefined && !!param.tfDeprecatedName) {
                  value = getFuncParam(
                      node.attr, param.tfDeprecatedName,
                      param.defaultValue as string);
                }
                break;
              case 'tensor':
              case 'tensors':
                break;
              default:
                throw new Error(
                    `Unsupported param type: ${param.type} for op: ${node.op}`);
            }
            map[param.name] = {value, type};
            return map;
          }, {});
    }
    return newNode;
  }

  // map the TFunctionDef to TFJS graph object
  private mapFunction(functionDef: tensorflow.IFunctionDef): Graph {
    const tfNodes = functionDef.nodeDef;
    const placeholders: Node[] = [];
    const weights: Node[] = [];
    let nodes: {[key: string]: Node} = {};
    if (tfNodes != null) {
      nodes = tfNodes.reduce<{[key: string]: Node}>((map, node) => {
        map[node.name] = this.mapNode(node);
        if (node.op === 'Const') {
          weights.push(map[node.name]);
        }
        return map;
      }, {});
    }
    const inputs: Node[] = [];
    const outputs: Node[] = [];

    functionDef.signature.inputArg.forEach(arg => {
      const [nodeName, ] = getNodeNameAndIndex(arg.name);
      const node: Node = {
        name: nodeName,
        op: 'Placeholder',
        inputs: [],
        inputNames: [],
        category: 'graph',
        inputParams: {},
        attrParams: {dtype: {value: parseDtypeParam(arg.type), type: 'dtype'}},
        children: []
      };
      node.signatureKey = arg.name;
      inputs.push(node);
      nodes[nodeName] = node;
    });

    const allNodes = Object.keys(nodes);
    allNodes.forEach(key => {
      const node = nodes[key];
      node.inputNames.forEach((name, index) => {
        const [nodeName, , outputName] = getNodeNameAndIndex(name);
        const inputNode = nodes[nodeName];
        if (inputNode.outputs != null) {
          const outputIndex = inputNode.outputs.indexOf(outputName);
          if (outputIndex !== -1) {
            const inputName = `${nodeName}:${outputIndex}`;
            // update the input name to use the mapped output index directly.
            node.inputNames[index] = inputName;
          }
        }
        node.inputs.push(inputNode);
        inputNode.children.push(node);
      });
    });

    const returnNodeMap = functionDef.ret;

    functionDef.signature.outputArg.forEach(output => {
      const [nodeName, index] = getNodeNameAndIndex(returnNodeMap[output.name]);
      const node = nodes[nodeName];
      if (node != null) {
        node.defaultOutput = index;
        outputs.push(node);
      }
    });

    const signature = this.mapArgsToSignature(functionDef);
    return {nodes, inputs, outputs, weights, placeholders, signature};
  }

  private mapArgsToSignature(functionDef: tensorflow.IFunctionDef):
      tensorflow.ISignatureDef {
    return {
      methodName: functionDef.signature.name,
      inputs: functionDef.signature.inputArg.reduce(
          (map, arg) => {
            map[arg.name] = this.mapArgToTensorInfo(arg);
            return map;
          },
          {} as {[key: string]: tensorflow.ITensorInfo}),
      outputs: functionDef.signature.outputArg.reduce(
          (map, arg) => {
            map[arg.name] = this.mapArgToTensorInfo(arg, functionDef.ret);
            return map;
          },
          {} as {[key: string]: tensorflow.ITensorInfo}),
    };
  }

  private mapArgToTensorInfo(
      arg: tensorflow.OpDef.IArgDef,
      nameMap?: {[key: string]: string}): tensorflow.ITensorInfo {
    let name = arg.name;
    if (nameMap != null) {
      name = nameMap[name];
    }
    return {name, dtype: arg.type};
  }
}

export function decodeBase64(text: string): string {
  const global = env().global;
  if (typeof global.atob !== 'undefined') {
    return global.atob(text);
  } else if (typeof Buffer !== 'undefined') {
    return new Buffer(text, 'base64').toString();
  } else {
    throw new Error(
        'Unable to decode base64 in this environment. ' +
        'Missing built-in atob() or Buffer()');
  }
}

export function parseStringParam(s: []|string, keepCase: boolean): string {
  const value =
      Array.isArray(s) ? String.fromCharCode.apply(null, s) : decodeBase64(s);
  return keepCase ? value : value.toLowerCase();
}

export function getStringParam(
    attrs: {[key: string]: tensorflow.IAttrValue}, name: string, def: string,
    keepCase = false): string {
  const param = attrs[name];
  if (param != null) {
    return parseStringParam(param.s, keepCase);
  }
  return def;
}

export function getBoolParam(
    attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
    def: boolean): boolean {
  const param = attrs[name];
  return param ? param.b : def;
}

export function getNumberParam(
    attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
    def: number): number {
  const param = attrs[name] || {};
  const value =
      param['i'] != null ? param['i'] : (param['f'] != null ? param['f'] : def);
  return (typeof value === 'number') ? value : parseInt(value, 10);
}

export function parseDtypeParam(value: string|tensorflow.DataType): DataType {
  if (typeof (value) === 'string') {
    // tslint:disable-next-line:no-any
    value = tensorflow.DataType[value as any];
  }
  switch (value) {
    case tensorflow.DataType.DT_FLOAT:
    case tensorflow.DataType.DT_HALF:
      return 'float32';
    case tensorflow.DataType.DT_INT32:
    case tensorflow.DataType.DT_INT64:
    case tensorflow.DataType.DT_INT8:
    case tensorflow.DataType.DT_UINT8:
      return 'int32';
    case tensorflow.DataType.DT_BOOL:
      return 'bool';
    case tensorflow.DataType.DT_DOUBLE:
      return 'float32';
    case tensorflow.DataType.DT_STRING:
      return 'string';
    case tensorflow.DataType.DT_COMPLEX64:
    case tensorflow.DataType.DT_COMPLEX128:
      return 'complex64';
    default:
      // Unknown dtype error will happen at runtime (instead of parse time),
      // since these nodes might not be used by the actual subgraph execution.
      return null;
  }
}

export function getFuncParam(
    attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
    def: string): string {
  const param = attrs[name];
  if (param && param.func) {
    return param.func.name;
  }
  return def;
}

export function getDtypeParam(
    attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
    def: DataType): DataType {
  const param = attrs[name];
  if (param && param.type) {
    return parseDtypeParam(param.type);
  }
  return def;
}

export function getDtypeArrayParam(
    attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
    def: DataType[]): DataType[] {
  const param = attrs[name];
  if (param && param.list && param.list.type) {
    return param.list.type.map(v => parseDtypeParam(v));
  }
  return def;
}

export function parseTensorShapeParam(shape: tensorflow.ITensorShape): number[]|
    undefined {
  if (shape.unknownRank) {
    return undefined;
  }
  if (shape.dim != null) {
    return shape.dim.map(
        dim =>
            (typeof dim.size === 'number') ? dim.size : parseInt(dim.size, 10));
  }
  return [];
}

export function getTensorShapeParam(
    attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
    def?: number[]): number[]|undefined {
  const param = attrs[name];
  if (param && param.shape) {
    return parseTensorShapeParam(param.shape);
  }
  return def;
}

export function getNumericArrayParam(
    attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
    def: number[]): number[] {
  const param = attrs[name];
  if (param) {
    return ((param.list.f && param.list.f.length ? param.list.f :
                                                   param.list.i) ||
            [])
        .map(v => (typeof v === 'number') ? v : parseInt(v, 10));
  }
  return def;
}

export function getStringArrayParam(
    attrs: {[key: string]: tensorflow.IAttrValue}, name: string, def: string[],
    keepCase = false): string[] {
  const param = attrs[name];
  if (param && param.list && param.list.s) {
    return param.list.s.map((v) => {
      return parseStringParam(v, keepCase);
    });
  }
  return def;
}

export function getTensorShapeArrayParam(
    attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
    def: number[][]): number[][] {
  const param = attrs[name];
  if (param && param.list && param.list.shape) {
    return param.list.shape.map((v) => {
      return parseTensorShapeParam(v);
    });
  }
  return def;
}

export function getBoolArrayParam(
    attrs: {[key: string]: tensorflow.IAttrValue}, name: string,
    def: boolean[]): boolean[] {
  const param = attrs[name];
  if (param && param.list && param.list.b) {
    return param.list.b;
  }
  return def;
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {DataType, Tensor} from '@tensorflow/tfjs-core';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {getTensor} from '../executors/utils';
import {getBoolArrayParam, getBoolParam, getDtypeArrayParam, getDtypeParam, getNumberParam, getNumericArrayParam, getStringArrayParam, getStringParam, getTensorShapeArrayParam, getTensorShapeParam} from '../operation_mapper';
import {GraphNode, Node, ValueType} from '../types';

/**
 * Helper class for lookup inputs and params for nodes in the model graph.
 */
export class NodeValueImpl implements GraphNode {
  public readonly inputs: Tensor[] = [];
  public readonly attrs: {[key: string]: ValueType} = {};
  constructor(
      private node: Node, private tensorMap: NamedTensorsMap,
      private context: ExecutionContext) {
    this.inputs = node.inputNames.map(name => this.getInput(name));
    if (node.rawAttrs != null) {
      this.attrs = Object.keys(node.rawAttrs)
                       .reduce((attrs: {[key: string]: ValueType}, key) => {
                         attrs[key] = this.getAttr(key);
                         return attrs;
                       }, {});
    }
  }

  /**
   * Return the value of the attribute or input param.
   * @param name String: name of attribute or input param.
   */
  private getInput(name: string): Tensor {
    return getTensor(name, this.tensorMap, this.context);
  }

  /**
   * Return the value of the attribute or input param.
   * @param name String: name of attribute or input param.
   */
  private getAttr(name: string, defaultValue?: ValueType): ValueType {
    const value = this.node.rawAttrs[name];
    if (value.tensor != null) {
      return getTensor(name, this.tensorMap, this.context);
    }
    if (value.i != null || value.f != null) {
      return getNumberParam(this.node.rawAttrs, name, defaultValue as number);
    }
    if (value.s != null) {
      return getStringParam(this.node.rawAttrs, name, defaultValue as string);
    }
    if (value.b != null) {
      return getBoolParam(this.node.rawAttrs, name, defaultValue as boolean);
    }
    if (value.shape != null) {
      return getTensorShapeParam(
          this.node.rawAttrs, name, defaultValue as number[]);
    }
    if (value.type != null) {
      return getDtypeParam(this.node.rawAttrs, name, defaultValue as DataType);
    }
    if (value.list != null) {
      if (value.list.i != null || value.list.f != null) {
        return getNumericArrayParam(
            this.node.rawAttrs, name, defaultValue as number[]);
      }
      if (value.list.s != null) {
        return getStringArrayParam(
            this.node.rawAttrs, name, defaultValue as string[]);
      }
      if (value.list.shape != null) {
        return getTensorShapeArrayParam(
            this.node.rawAttrs, name, defaultValue as number[][]);
      }
      if (value.list.b != null) {
        return getBoolArrayParam(
            this.node.rawAttrs, name, defaultValue as boolean[]);
      }
      if (value.list.type != null) {
        return getDtypeArrayParam(
            this.node.rawAttrs, name, defaultValue as DataType[]);
      }
    }

    return defaultValue;
  }
}


/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {OpExecutor, OpMapper} from '../types';

const CUSTOM_OPS: {[key: string]: OpMapper} = {};

/**
 * Register an Op for graph model executor. This allows you to register
 * TensorFlow custom op or override existing op.
 *
 * Here is an example of registering a new MatMul Op.
 * ```js
 * const customMatmul = (node) =>
 *    tf.matMul(
 *        node.inputs[0], node.inputs[1],
 *        node.attrs['transpose_a'], node.attrs['transpose_b']);
 *
 * tf.registerOp('MatMul', customMatmul);
 * ```
 * The inputs and attrs of the node object are based on the TensorFlow op
 * registry.
 *
 * @param name The Tensorflow Op name.
 * @param opFunc An op function which is called with the current graph node
 * during execution and needs to return a tensor or a list of tensors. The node
 * has the following attributes:
 *    - attr: A map from attribute name to its value
 *    - inputs: A list of input tensors
 *
 * @doc {heading: 'Models', subheading: 'Op Registry'}
 */
export function registerOp(name: string, opFunc: OpExecutor) {
  const opMapper: OpMapper = {
    tfOpName: name,
    category: 'custom',
    inputs: [],
    attrs: [],
    customExecutor: opFunc
  };

  CUSTOM_OPS[name] = opMapper;
}

/**
 * Retrieve the OpMapper object for the registered op.
 *
 * @param name The Tensorflow Op name.
 *
 * @doc {heading: 'Models', subheading: 'Op Registry'}
 */
export function getRegisteredOp(name: string): OpMapper {
  return CUSTOM_OPS[name];
}

/**
 * Deregister the Op for graph model executor.
 *
 * @param name The Tensorflow Op name.
 *
 * @doc {heading: 'Models', subheading: 'Op Registry'}
 */
export function deregisterOp(name: string) {
  delete CUSTOM_OPS[name];
}

/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor, Tensor1D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
     ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'RaggedGather': {
          const {
            outputNestedSplits,
            outputDenseValues,
          } =
              ops.raggedGather(
                  getParamValue(
                      'paramsNestedSplits', node, tensorMap, context) as
                      Tensor[],
                  getParamValue(
                      'paramsDenseValues', node, tensorMap, context) as Tensor,
                  getParamValue('indices', node, tensorMap, context) as Tensor,
                  getParamValue('outputRaggedRank', node, tensorMap, context) as
                      number);
          return outputNestedSplits.concat(outputDenseValues);
        }
        case 'RaggedRange': {
          const {rtNestedSplits, rtDenseValues} = ops.raggedRange(
              getParamValue('starts', node, tensorMap, context) as Tensor,
              getParamValue('limits', node, tensorMap, context) as Tensor,
              getParamValue('splits', node, tensorMap, context) as Tensor);
          return [rtNestedSplits, rtDenseValues];
        }
        case 'RaggedTensorToTensor': {
          return [ops.raggedTensorToTensor(
              getParamValue('shape', node, tensorMap, context) as Tensor,
              getParamValue('values', node, tensorMap, context) as Tensor1D,
              getParamValue('defaultValue', node, tensorMap, context) as Tensor,
              getParamValue('rowPartitionTensors', node, tensorMap, context) as
                  Tensor[],
              getParamValue('rowPartitionTypes', node, tensorMap, context) as
                  string[])];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'ragged';

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Scalar, Tensor, Tensor1D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext, ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'StaticRegexReplace': {
          return [ops.string.staticRegexReplace(
            getParamValue('input', node, tensorMap, context) as Tensor,
            getParamValue('pattern', node, tensorMap, context) as string,
            getParamValue('rewrite', node, tensorMap, context) as string,
            getParamValue('replaceGlobal', node, tensorMap, context) as boolean,
          )];
        }
        case 'StringNGrams': {
          const {nGrams, nGramsSplits} = ops.string.stringNGrams(
              getParamValue('data', node, tensorMap, context) as Tensor1D,
              getParamValue('dataSplits', node, tensorMap, context) as Tensor,
              getParamValue('separator', node, tensorMap, context) as string,
              getParamValue('nGramWidths', node, tensorMap, context) as
                  number[],
              getParamValue('leftPad', node, tensorMap, context) as string,
              getParamValue('rightPad', node, tensorMap, context) as string,
              getParamValue('padWidth', node, tensorMap, context) as number,
              getParamValue(
                  'preserveShortSequences', node, tensorMap, context) as
                  boolean);
          return [nGrams, nGramsSplits];
        }
        case 'StringSplit': {
          const {indices, values, shape} = ops.string.stringSplit(
              getParamValue('input', node, tensorMap, context) as Tensor1D,
              getParamValue('delimiter', node, tensorMap, context) as Scalar,
              getParamValue('skipEmpty', node, tensorMap, context) as boolean);
          return [indices, values, shape];
        }
        case 'StringToHashBucketFast': {
          const output = ops.string.stringToHashBucketFast(
              getParamValue('input', node, tensorMap, context) as Tensor,
              getParamValue('numBuckets', node, tensorMap, context) as number);
          return [output];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'string';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue, getTensor} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
     ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'Abs':
        case 'ComplexAbs':
          return [ops.abs(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Acos':
          return [ops.acos(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Acosh':
          return [ops.acosh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Asin':
          return [ops.asin(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Asinh':
          return [ops.asinh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Atan':
          return [ops.atan(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Atan2':
          return [ops.atan2(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('y', node, tensorMap, context) as Tensor)];
        case 'Atanh':
          return [ops.atanh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Ceil':
          return [ops.ceil(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Complex':
          return [ops.complex(
              getParamValue('real', node, tensorMap, context) as Tensor,
              getParamValue('imag', node, tensorMap, context) as Tensor)];
        case 'Cos':
          return [ops.cos(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Cosh':
          return [ops.cosh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Elu':
          return [ops.elu(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Erf':
          return [ops.erf(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Exp':
          return [ops.exp(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Expm1': {
          return [ops.expm1(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Floor':
          return [ops.floor(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Log':
          return [ops.log(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Log1p': {
          return [ops.log1p(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Imag':
          return [ops.imag(
              getParamValue('x', node, tensorMap, context) as Tensor)];

        case 'Neg':
          return [ops.neg(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Reciprocal': {
          return [ops.reciprocal(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Real':
          return [ops.real(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Relu':
          return [ops.relu(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Round': {
          return [ops.round(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Selu':
          return [ops.selu(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Sigmoid':
          return [ops.sigmoid(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Sin':
          return [ops.sin(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Sign': {
          return [ops.sign(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Sinh': {
          return [ops.sinh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Softplus': {
          return [ops.softplus(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Sqrt': {
          return [ops.sqrt(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Square': {
          return [ops.square(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Tanh': {
          return [ops.tanh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Tan':
          return [ops.tan(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'ClipByValue':
          return [ops.clipByValue(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('clipValueMin', node, tensorMap, context) as number,
              getParamValue('clipValueMax', node, tensorMap, context) as
                  number)];
        case 'Relu6':
          return [ops.relu6(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Rsqrt':
          return [ops.rsqrt(getTensor(node.inputNames[0], tensorMap, context))];
        case 'LeakyRelu':
          return [ops.leakyRelu(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('alpha', node, tensorMap, context) as number)];
        case 'Prelu':
          return [ops.prelu(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('alpha', node, tensorMap, context) as Tensor)];
        case 'IsNan':
          return [ops.isNaN(getTensor(node.inputNames[0], tensorMap, context))];
        case 'IsInf':
          return [ops.isInf(getTensor(node.inputNames[0], tensorMap, context))];
        case 'IsFinite':
          return [ops.isFinite(
              getTensor(node.inputNames[0], tensorMap, context))];
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'basic_math';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
     ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'Equal': {
          return [ops.equal(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'NotEqual': {
          return [ops.notEqual(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'Greater': {
          return [ops.greater(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'GreaterEqual': {
          return [ops.greaterEqual(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'Less': {
          return [ops.less(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'LessEqual': {
          return [ops.lessEqual(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'LogicalAnd': {
          return [ops.logicalAnd(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'LogicalNot': {
          return [ops.logicalNot(
              getParamValue('a', node, tensorMap, context) as Tensor)];
        }
        case 'LogicalOr': {
          return [ops.logicalOr(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'Select':
        case 'SelectV2': {
          return [ops.where(
              getParamValue('condition', node, tensorMap, context) as Tensor,
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'BitwiseAnd': {
          return [ops.bitwiseAnd(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'logical';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
     ops = tfOps): Tensor[] => {
          switch (node.op) {
            case 'FFT': {
              return [ops.fft(
                  getParamValue('x', node, tensorMap, context) as Tensor)];
            }
            case 'IFFT': {
              return [ops.ifft(
                  getParamValue('x', node, tensorMap, context) as Tensor)];
            }
            case 'RFFT': {
              return [ops.rfft(
                  getParamValue('x', node, tensorMap, context) as Tensor)];
            }
            case 'IRFFT': {
              return [ops.irfft(
                  getParamValue('x', node, tensorMap, context) as Tensor)];
            }
            default:
              throw TypeError(`Node type ${node.op} is not implemented`);
          }
        };

export const CATEGORY = 'spectral';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext, ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'BiasAdd':
        case 'AddV2':
        case 'Add': {
          return [ops.add(
              (getParamValue('a', node, tensorMap, context) as Tensor),
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'AddN': {
          return [ops.addN((
              getParamValue('tensors', node, tensorMap, context) as Tensor[]))];
        }
        case 'FloorMod':
        case 'Mod':
          return [ops.mod(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        case 'Mul':
          return [ops.mul(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        case 'RealDiv':
        case 'Div': {
          return [ops.div(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'DivNoNan': {
          return [ops.divNoNan(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'FloorDiv': {
          return [ops.floorDiv(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'Sub': {
          return [ops.sub(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'Minimum': {
          return [ops.minimum(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'Maximum': {
          return [ops.maximum(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'Pow': {
          return [ops.pow(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'SquaredDifference': {
          return [ops.squaredDifference(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'arithmetic';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {InputParamValue, OpMapper, ParamValue} from '../types';
import {Node} from '../types';

export function createNumberAttr(value: number): ParamValue {
  return {value, type: 'number'};
}

export function createNumberAttrFromIndex(inputIndex: number): InputParamValue {
  return {inputIndexStart: inputIndex, type: 'number'};
}

export function createStrAttr(str: string): ParamValue {
  return {value: str, type: 'string'};
}

export function createStrArrayAttr(strs: string[]): ParamValue {
  return {value: strs, type: 'string[]'};
}

export function createBoolAttr(value: boolean): ParamValue {
  return {value, type: 'bool'};
}

export function createTensorShapeAttr(value: number[]): ParamValue {
  return {value, type: 'shape'};
}

export function createShapeAttrFromIndex(inputIndex: number): InputParamValue {
  return {inputIndexStart: inputIndex, type: 'shape'};
}

export function createNumericArrayAttr(value: number[]): ParamValue {
  return {value, type: 'number[]'};
}

export function createNumericArrayAttrFromIndex(inputIndex: number):
    InputParamValue {
  return {inputIndexStart: inputIndex, type: 'number[]'};
}

export function createBooleanArrayAttrFromIndex(inputIndex: number):
    InputParamValue {
  return {inputIndexStart: inputIndex, type: 'bool[]'};
}

export function createTensorAttr(index: number): InputParamValue {
  return {inputIndexStart: index, type: 'tensor'};
}

export function createTensorsAttr(
    index: number, paramLength: number): InputParamValue {
  return {inputIndexStart: index, inputIndexEnd: paramLength, type: 'tensors'};
}

export function createDtypeAttr(dtype: string): ParamValue {
  return {value: dtype, type: 'dtype'};
}

export function validateParam(
    node: Node, opMappers: OpMapper[], tfOpName?: string) {
  const opMapper = tfOpName != null ?
      opMappers.find(mapper => mapper.tfOpName === tfOpName) :
      opMappers.find(mapper => mapper.tfOpName === node.op);
  const matched = Object.keys(node.inputParams).every(key => {
    const value = node.inputParams[key];
    const def = opMapper.inputs.find(param => param.name === key);
    return def && def.type === value.type &&
        def.start === value.inputIndexStart && def.end === value.inputIndexEnd;
  }) &&
      Object.keys(node.attrParams).every(key => {
        const value = node.attrParams[key];
        const def = opMapper.attrs.find(param => param.name === key);
        return def && def.type === value.type;
      });
  if (!matched) {
    console.log('node = ', node);
    console.log('opMapper = ', opMapper);
  }
  return matched;
}

export function uncapitalize<Name extends string>(name: Name): Uncapitalize<Name> {
  return name.charAt(0).toLowerCase() + name.slice(1) as Uncapitalize<Name>;
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {clone, Tensor, util} from '@tensorflow/tfjs-core';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {ResourceManager} from '../../executor/resource_manager';
import {Node, ValueType} from '../types';

export function getParamValue(
    paramName: string, node: Node, tensorMap: NamedTensorsMap,
    context: ExecutionContext, resourceManager?: ResourceManager): ValueType {
  const inputParam = node.inputParams[paramName];
  if (inputParam && inputParam.inputIndexStart !== undefined) {
    const start = inputParam.inputIndexStart;
    const end = inputParam.inputIndexEnd === 0 ?
        undefined :
        (inputParam.inputIndexEnd === undefined ? start + 1 :
                                                  inputParam.inputIndexEnd);
    const shiftedStart = start < 0 ? node.inputNames.length + start : start;
    if (inputParam.type === 'tensor') {
      return getTensor(
          node.inputNames[shiftedStart], tensorMap, context, resourceManager);
    }
    if (inputParam.type === 'tensors') {
      // TODO(mattSoulanille): This filters out NoOp nodes during execution, but
      // these should really never be in the execution graph in the first place.
      // They're necessary for ordering the graph, but should not be visible
      // during execution. Perhaps have different sets of children, one for
      // control dependencies and another for real dependencies.
      const inputs = node.inputs.slice(start, end);
      const inputNames = node.inputNames.slice(start, end)
        .filter((_name, index) => inputs[index]?.op !== 'NoOp');

      return inputNames.map(
          name => getTensor(name, tensorMap, context, resourceManager));
    }
    const tensor = getTensor(
        node.inputNames[shiftedStart], tensorMap, context, resourceManager);
    const data = tensor.dataSync();
    return inputParam.type === 'number' ?
        data[0] :
        util.toNestedArray(tensor.shape, data);
  }
  const attrParam = node.attrParams[paramName];
  return attrParam && attrParam.value;
}

/**
 * Retrieve the tensor from tensorsMap based on input name.
 * @param name Node input name
 * @param tensorsMap Tensors map keyed by the node
 * @param context contains tensors and information for running the current node.
 * @param resourceManager Optional. Contains global resources of the model.
 */
export function getTensor(
    name: string, tensorsMap: NamedTensorsMap, context: ExecutionContext,
    resourceManager?: ResourceManager): Tensor {
  const [nodeName, index] = parseNodeName(name, context);

  if (resourceManager != null) {
    const tensor = resourceManager.getHashTableHandleByName(nodeName);
    if (tensor != null) {
      return tensor;
    }
  }

  const contextId = context.currentContextIds.find(contextId => {
    return !!tensorsMap[getNodeNameWithContextId(nodeName, contextId)];
  });

  return contextId !== undefined ?
      tensorsMap[getNodeNameWithContextId(nodeName, contextId)][index] :
      undefined;
}

/**
 * Retrieve the tensors based on input name for current context.
 * @param name Node input name
 * @param tensorsMap Tensors map keyed by the node
 */
export function getTensorsForCurrentContext(
    name: string, tensorsMap: NamedTensorsMap,
    context: ExecutionContext): Tensor[] {
  return tensorsMap[getNodeNameWithContextId(name, context.currentContextId)];
}

/**
 * Returns the node name, outputName and index from the Node input name.
 * @param inputName The input name of the node, in format of
 * node_name:output_index, i.e. MatMul:0, if the output_index is not set, it is
 * default to 0.
 * If the input name contains output name i.e. StringSplit:indices:0, it will
 * return ['StringSplit', 0, 'indices'].
 */
export function getNodeNameAndIndex(
    inputName: string, context?: ExecutionContext): [string, number, string] {
  const [nodeName, index, outputName] = parseNodeName(inputName, context);

  return [
    getNodeNameWithContextId(nodeName, context && context.currentContextId),
    index, outputName
  ];
}

function getNodeNameWithContextId(name: string, contextId?: string): string {
  return !!contextId ? `${name}-${contextId}` : name;
}

export function parseNodeName(
    name: string, context?: ExecutionContext): [string, number, string?] {
  if (name === '') {
    return ['', 0, undefined];
  }

  const isCacheEnabled = context != null && context.parseNodeNameCache != null;
  if (isCacheEnabled) {
    const cachedResult = context.parseNodeNameCache.get(name);
    if (cachedResult != null) {
      return cachedResult;
    }
  }
  const parts = name.split(':');
  let result: [string, number, string?];
  if (parts.length === 1) {
    result = [name, 0, undefined];
  } else {
    const nodeName = parts[0];
    const outputName = parts.length === 3 ? parts[1] : undefined;
    const index = Number(parts[parts.length - 1]);
    result = [nodeName, index, outputName];
  }
  if (isCacheEnabled) {
    context.parseNodeNameCache.set(name, result);
  }
  return result;
}

export function split(arr: number[], size: number) {
  const res = [];
  for (let i = 0; i < arr.length; i += size) {
    res.push(arr.slice(i, i + size));
  }
  return res;
}
export function getPadding(
    node: Node, tensorMap: NamedTensorsMap,
    context: ExecutionContext): ValueType {
  let pad = getParamValue('pad', node, tensorMap, context);
  if (pad === 'explicit') {
    // This is 1d array, we need to convert it to 2d array
    pad = getParamValue('explicitPaddings', node, tensorMap, context);
    const explicitPadding: [
      [number, number], [number, number], [number, number], [number, number]
    ] = [[0, 0], [0, 0], [0, 0], [0, 0]];
    for (let i = 0; i < 4; i++) {
      explicitPadding[i][0] = (pad as number[])[i * 2];
      explicitPadding[i][1] = (pad as number[])[i * 2 + 1];
    }
    return explicitPadding;
  }
  return pad;
}

/**
 *  Reuse the tensor if it is marked as keep, otherwise clone the tensor to
 *  avoid disposal. This is important for TensorArray and TensorList ops, since
 *  internally they use a tensor as the id for TensorArray and TensorList, and
 * to simplify lookup, they also use Tensor.id as the key to the internal map.
 * These id tensors have been marked as kept in the backend, we need avoid clone
 * them in order to create new Tensor.id.
 * @param tensor
 */
export function cloneTensor(tensor: Tensor): Tensor {
  return tensor.kept ? tensor : clone(tensor);
}

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Scalar, Tensor, Tensor1D, Tensor2D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext, ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'SparseFillEmptyRows': {
          const {
            outputIndices,
            outputValues,
            emptyRowIndicator,
            reverseIndexMap
          } =
              ops.sparse.sparseFillEmptyRows(
                  getParamValue('indices', node, tensorMap, context) as
                      Tensor2D,
                  getParamValue('values', node, tensorMap, context) as Tensor1D,
                  getParamValue('denseShape', node, tensorMap, context) as
                      Tensor1D,
                  getParamValue('defaultValue', node, tensorMap, context) as
                      Scalar);
          return [
            outputIndices, outputValues, emptyRowIndicator, reverseIndexMap
          ];
        }
        case 'SparseReshape': {
          const {outputIndices, outputShape} = ops.sparse.sparseReshape(
              getParamValue('inputIndices', node, tensorMap, context) as
                  Tensor2D,
              getParamValue('inputShape', node, tensorMap, context) as Tensor1D,
              getParamValue('newShape', node, tensorMap, context) as Tensor1D);
          return [outputIndices, outputShape];
        }
        case 'SparseSegmentMean': {
          const outputData = ops.sparse.sparseSegmentMean(
              getParamValue('data', node, tensorMap, context) as Tensor,
              getParamValue('indices', node, tensorMap, context) as Tensor1D,
              getParamValue('segmentIds', node, tensorMap, context) as
                  Tensor1D);
          return [outputData];
        }
        case 'SparseSegmentSum': {
          const outputData = ops.sparse.sparseSegmentSum(
              getParamValue('data', node, tensorMap, context) as Tensor,
              getParamValue('indices', node, tensorMap, context) as Tensor1D,
              getParamValue('segmentIds', node, tensorMap, context) as
                  Tensor1D);
          return [outputData];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'sparse';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Scalar, Tensor, Tensor2D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
     ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'BatchMatMul':
        case 'BatchMatMulV2':
        case 'MatMul':
          return [ops.matMul(
              getParamValue('a', node, tensorMap, context) as Tensor2D,
              getParamValue('b', node, tensorMap, context) as Tensor2D,
              getParamValue('transposeA', node, tensorMap, context) as boolean,
              getParamValue('transposeB', node, tensorMap, context) as
                  boolean)];

        case 'Einsum':
          return [ops.einsum(
              getParamValue('equation', node, tensorMap, context) as string,
              ...getParamValue('tensors', node, tensorMap, context) as
                  Tensor[])];

        case 'Transpose':
          return [ops.transpose(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('perm', node, tensorMap, context) as number[])];

        case '_FusedMatMul':
          const [extraOp, activationFunc] =
              (getParamValue('fusedOps', node, tensorMap, context) as string[]);

          const isBiasAdd = extraOp === 'biasadd';
          const isPrelu = activationFunc === 'prelu';

          const numArgs =
              (getParamValue('numArgs', node, tensorMap, context) as number);
          const leakyreluAlpha =
              getParamValue('leakyreluAlpha', node, tensorMap, context) as
              number;

          if (isBiasAdd) {
            if (isPrelu && numArgs !== 2) {
              throw new Error(
                  'Fused MatMul with BiasAdd and Prelu must have two ' +
                  'extra arguments: bias and alpha.');
            }
            if (!isPrelu && numArgs !== 1) {
              throw new Error(
                  'Fused MatMul with BiasAdd must have one extra argument: bias.');
            }
          }
          const [biasArg, preluArg] =
              getParamValue('args', node, tensorMap, context) as Tensor[];
          return [ops.fused.matMul({
            a: getParamValue('a', node, tensorMap, context) as Tensor2D,
            b: getParamValue('b', node, tensorMap, context) as Tensor2D,
            transposeA: getParamValue('transposeA', node, tensorMap, context) as
                boolean,
            transposeB: getParamValue('transposeB', node, tensorMap, context) as
                boolean,
            bias: biasArg,
            activation: activationFunc as tfOps.fused.Activation,
            preluActivationWeights: preluArg,
            leakyreluAlpha
          })];

        case 'MatrixBandPart':
          return [ops.linalg.bandPart(
              getParamValue('a', node, tensorMap, context) as Tensor2D,
              getParamValue('numLower', node, tensorMap, context) as Scalar,
              getParamValue('numUpper', node, tensorMap, context) as Scalar)];

        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'matrices';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {DataType, Tensor, Tensor1D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
     ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'Fill': {
          const shape =
              getParamValue('shape', node, tensorMap, context) as number[];
          const dtype =
              getParamValue('dtype', node, tensorMap, context) as DataType;
          const value =
              getParamValue('value', node, tensorMap, context) as number;
          return [ops.fill(shape, value, dtype)];
        }
        case 'LinSpace': {
          const start =
              getParamValue('start', node, tensorMap, context) as number;
          const stop =
              getParamValue('stop', node, tensorMap, context) as number;
          const num = getParamValue('num', node, tensorMap, context) as number;
          return [ops.linspace(start, stop, num)];
        }
        case 'Multinomial': {
          const logits =
              getParamValue('logits', node, tensorMap, context) as Tensor1D;
          const numSamples =
              getParamValue('numSamples', node, tensorMap, context) as number;
          const seed =
              getParamValue('seed', node, tensorMap, context) as number;
          return [ops.multinomial(logits, numSamples, seed)];
        }
        case 'OneHot': {
          const indices =
              getParamValue('indices', node, tensorMap, context) as Tensor1D;
          const depth =
              getParamValue('depth', node, tensorMap, context) as number;
          const onValue =
              getParamValue('onValue', node, tensorMap, context) as number;
          const offValue =
              getParamValue('offValue', node, tensorMap, context) as number;
          const dtype =
              getParamValue('dtype', node, tensorMap, context) as DataType;
          return [ops.oneHot(indices, depth, onValue, offValue, dtype)];
        }
        case 'Ones': {
          return [ops.ones(
              getParamValue('shape', node, tensorMap, context) as number[],
              getParamValue('dtype', node, tensorMap, context) as DataType)];
        }
        case 'OnesLike': {
          return [ops.onesLike(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'RandomStandardNormal': {
          return [ops.randomStandardNormal(
              getParamValue('shape', node, tensorMap, context) as number[],
              getParamValue('dtype', node, tensorMap, context) as 'float32' |
                  'int32',
              getParamValue('seed', node, tensorMap, context) as number)];
        }
        case 'RandomUniform': {
          return [ops.randomUniform(
              // tslint:disable-next-line:no-any
              getParamValue('shape', node, tensorMap, context) as any,
              getParamValue('minval', node, tensorMap, context) as number,
              getParamValue('maxval', node, tensorMap, context) as number,
              getParamValue('dtype', node, tensorMap, context) as DataType)];
        }
        case 'RandomUniformInt': {
          return [ops.randomUniformInt(
              getParamValue('shape', node, tensorMap, context) as number[],
              getParamValue('minval', node, tensorMap, context) as number,
              getParamValue('maxval', node, tensorMap, context) as number,
              getParamValue('seed', node, tensorMap, context) as number)];
        }
        case 'Range': {
          const start =
              getParamValue('start', node, tensorMap, context) as number;
          const stop =
              getParamValue('stop', node, tensorMap, context) as number;
          const step =
              getParamValue('step', node, tensorMap, context) as number;
          return [ops.range(
              start, stop, step,
              getParamValue('dtype', node, tensorMap, context) as 'float32' |
                  'int32')];
        }
        case 'TruncatedNormal': {
          const shape =
              getParamValue('shape', node, tensorMap, context) as number[];
          const mean =
              getParamValue('mean', node, tensorMap, context) as number;
          const stdDev =
              getParamValue('stdDev', node, tensorMap, context) as number;
          const seed =
              getParamValue('seed', node, tensorMap, context) as number;
          return [ops.truncatedNormal(
              shape, mean, stdDev,
              getParamValue('dtype', node, tensorMap, context) as 'float32' |
                  'int32',
              seed)];
        }
        case 'Zeros': {
          return [ops.zeros(
              getParamValue('shape', node, tensorMap, context) as number[],
              getParamValue('dtype', node, tensorMap, context) as DataType)];
        }
        case 'ZerosLike': {
          return [ops.zerosLike(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'creation';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor, Tensor1D, Tensor2D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import { ResourceManager } from '../../executor/resource_manager';
import {InternalOpAsyncExecutor, Node} from '../types';

import {getParamValue} from './utils';

function nmsParams(
    node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext) {
  const boxes = getParamValue('boxes', node, tensorMap, context) as Tensor;
  const scores = getParamValue('scores', node, tensorMap, context) as Tensor;
  const maxOutputSize =
      getParamValue('maxOutputSize', node, tensorMap, context) as number;
  const iouThreshold =
      getParamValue('iouThreshold', node, tensorMap, context) as number;
  const scoreThreshold =
      getParamValue('scoreThreshold', node, tensorMap, context) as number;
  const softNmsSigma =
      getParamValue('softNmsSigma', node, tensorMap, context) as number;

  return {
    boxes,
    scores,
    maxOutputSize,
    iouThreshold,
    scoreThreshold,
    softNmsSigma
  };
}

export const executeOp: InternalOpAsyncExecutor = async(
    node: Node, tensorMap: NamedTensorsMap,
    context: ExecutionContext, resourceManager: ResourceManager,
    ops = tfOps): Promise<Tensor[]> => {
  switch (node.op) {
    case 'NonMaxSuppressionV5': {
      const {
        boxes,
        scores,
        maxOutputSize,
        iouThreshold,
        scoreThreshold,
        softNmsSigma
      } = nmsParams(node, tensorMap, context);

      const result = await ops.image.nonMaxSuppressionWithScoreAsync(
          boxes as Tensor2D, scores as Tensor1D, maxOutputSize, iouThreshold,
          scoreThreshold, softNmsSigma);

      return [result.selectedIndices, result.selectedScores];
    }
    case 'NonMaxSuppressionV4': {
      const {boxes, scores, maxOutputSize, iouThreshold, scoreThreshold} =
          nmsParams(node, tensorMap, context);

      const padToMaxOutputSize =
          getParamValue('padToMaxOutputSize', node, tensorMap, context) as
          boolean;

      const result = await ops.image.nonMaxSuppressionPaddedAsync(
          boxes as Tensor2D, scores as Tensor1D, maxOutputSize, iouThreshold,
          scoreThreshold, padToMaxOutputSize);

      return [result.selectedIndices, result.validOutputs];
    }
    case 'NonMaxSuppressionV3':
    case 'NonMaxSuppressionV2': {
      const {boxes, scores, maxOutputSize, iouThreshold, scoreThreshold} =
          nmsParams(node, tensorMap, context);

      return [await ops.image.nonMaxSuppressionAsync(
          boxes as Tensor2D, scores as Tensor1D, maxOutputSize, iouThreshold,
          scoreThreshold)];
    }
    case 'Where': {
      const condition = ops.cast(
          (getParamValue('condition', node, tensorMap, context) as Tensor),
          'bool');
      const result = [await ops.whereAsync(condition)];
      condition.dispose();
      return result;
    }
    case 'ListDiff': {
      return ops.setdiff1dAsync(
          getParamValue('x', node, tensorMap, context) as Tensor,
          getParamValue('y', node, tensorMap, context) as Tensor);
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'dynamic';

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {DataType, Tensor} from '@tensorflow/tfjs-core';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {HashTable} from '../../executor/hash_table';
import {ResourceManager} from '../../executor/resource_manager';
import {InternalOpAsyncExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpAsyncExecutor = async(
    node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
    resourceManager: ResourceManager): Promise<Tensor[]> => {
  switch (node.op) {
    case 'HashTable':
    case 'HashTableV2': {
      const existingTableHandle =
          resourceManager.getHashTableHandleByName(node.name);
      // Table is shared with initializer.
      if (existingTableHandle != null) {
        return [existingTableHandle];
      } else {
        const keyDType =
            getParamValue('keyDType', node, tensorMap, context) as DataType;
        const valueDType =
            getParamValue('valueDType', node, tensorMap, context) as DataType;

        const hashTable = new HashTable(keyDType, valueDType);
        resourceManager.addHashTable(node.name, hashTable);
        return [hashTable.handle];
      }
    }
    case 'InitializeTable':
    case 'InitializeTableV2':
    case 'LookupTableImport':
    case 'LookupTableImportV2': {
      const handle = getParamValue(
                         'tableHandle', node, tensorMap, context,
                         resourceManager) as Tensor;
      const keys = getParamValue('keys', node, tensorMap, context) as Tensor;
      const values =
          getParamValue('values', node, tensorMap, context) as Tensor;

      const hashTable = resourceManager.getHashTableById(handle.id);

      return [await hashTable.import(keys, values)];
    }
    case 'LookupTableFind':
    case 'LookupTableFindV2': {
      const handle = getParamValue(
                         'tableHandle', node, tensorMap, context,
                         resourceManager) as Tensor;
      const keys = getParamValue('keys', node, tensorMap, context) as Tensor;
      const defaultValue =
          getParamValue('defaultValue', node, tensorMap, context) as Tensor;

      const hashTable = resourceManager.getHashTableById(handle.id);
      return [await hashTable.find(keys, defaultValue)];
    }
    case 'LookupTableSize':
    case 'LookupTableSizeV2': {
      const handle = getParamValue(
                         'tableHandle', node, tensorMap, context,
                         resourceManager) as Tensor;

      const hashTable = resourceManager.getHashTableById(handle.id);
      return [hashTable.tensorSize()];
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'hash_table';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {cloneTensor, getParamValue, getTensor} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext, ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'Const': {
          return tensorMap[node.name];
        }
        case 'PlaceholderWithDefault':
          const def =
              getParamValue('default', node, tensorMap, context) as Tensor;
          return [getTensor(node.name, tensorMap, context) || def];
        case 'Placeholder':
          return [getTensor(node.name, tensorMap, context)];
        case 'Identity':
        case 'StopGradient':
        case 'FakeQuantWithMinMaxVars': {  // This op is currently ignored.
          const data = getParamValue('x', node, tensorMap, context) as Tensor;
          return [cloneTensor(data)];
        }
        case 'IdentityN':
          return (getParamValue('x', node, tensorMap, context) as Tensor[])
              .map((t: Tensor) => cloneTensor(t));
        case 'Snapshot':
          const snapshot =
              (getParamValue('x', node, tensorMap, context) as Tensor);
          return [cloneTensor(snapshot)];
        case 'Shape':
          return [ops.tensor1d(
              (getParamValue('x', node, tensorMap, context) as Tensor).shape,
              'int32')];
        case 'ShapeN':
          return (getParamValue('x', node, tensorMap, context) as Tensor[])
              .map((t: Tensor) => ops.tensor1d(t.shape));
        case 'Size':
          return [ops.scalar(
              (getParamValue('x', node, tensorMap, context) as Tensor).size,
              'int32')];
        case 'Rank':
          return [ops.scalar(
              (getParamValue('x', node, tensorMap, context) as Tensor).rank,
              'int32')];
        case 'NoOp':
          return [ops.scalar(1)];
        case 'Print':
          const input = getParamValue('x', node, tensorMap, context) as Tensor;
          const data =
              getParamValue('data', node, tensorMap, context) as Tensor[];
          const message =
              getParamValue('message', node, tensorMap, context) as string;
          const summarize =
              getParamValue('summarize', node, tensorMap, context) as number;
          console.warn(
              'The graph has a tf.print() operation,' +
              'usually used for debugging, which slows down performance.');
          console.log(message);
          for (let i = 0; i < data.length; i++) {
            console.log(Array.prototype.slice.call(data[i].dataSync())
                            .slice(0, summarize));
          }
          return [input];

        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'graph';

/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// The opposite of Extract<T, U>
type Without<T, U> = T extends U ? never : T;

// Do not spy on CompositeArrayBuffer because it is a class constructor.
type NotSpiedOn = 'CompositeArrayBuffer';

export type RecursiveSpy<T> =
  T extends Function ? jasmine.Spy :
  {[K in Without<keyof T, NotSpiedOn>]: RecursiveSpy<T[K]>} &
  {[K in Extract<keyof T, NotSpiedOn>]: T[K]};

export function spyOnAllFunctions<T>(obj: T): RecursiveSpy<T> {
  return Object.fromEntries(Object.entries(obj).map(([key, val]) => {
    // TODO(mattSoulanille): Do not hard code this
    if (key === 'CompositeArrayBuffer') {
      return val;
    }
    if (val instanceof Function) {
      return [key, jasmine.createSpy(`${key} spy`, val).and.callThrough()];
    } else if (val instanceof Array) {
      return [key, val];
    } else if (val instanceof Object) {
      return [key, spyOnAllFunctions(val)];
    }
    return [key, val];
  })) as RecursiveSpy<T>;
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Rank, Tensor, Tensor3D, Tensor4D, Tensor5D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getPadding, getParamValue} from './utils';

function fusedConvAndDepthWiseParams(
    node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext) {
  const [extraOp, activationFunc] =
      (getParamValue('fusedOps', node, tensorMap, context) as string[]);

  const isBiasAdd = extraOp === 'biasadd';
  const noBiasAdd = !isBiasAdd;
  const isPrelu = activationFunc === 'prelu';
  const isBatchNorm = extraOp === 'fusedbatchnorm';

  const numArgs =
      (getParamValue('numArgs', node, tensorMap, context) as number);
  if (isBiasAdd) {
    if (isPrelu && numArgs !== 2) {
      throw new Error(
          'FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu ' +
          'must have two extra arguments: bias and alpha.');
    }
    if (!isPrelu && isBiasAdd && numArgs !== 1) {
      throw new Error(
          'FusedConv2d and DepthwiseConv2d with BiasAdd must have ' +
          'one extra argument: bias.');
    }
  }
  if (isBatchNorm) {
    throw new Error(
        'FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported');
  }
  const stride = getParamValue('strides', node, tensorMap, context) as number[];
  const pad = getPadding(node, tensorMap, context);
  const dataFormat =
      (getParamValue('dataFormat', node, tensorMap, context) as string)
          .toUpperCase();
  const dilations =
      getParamValue('dilations', node, tensorMap, context) as number[];
  let [biasArg, preluArg] =
      getParamValue('args', node, tensorMap, context) as Tensor[];
  if (noBiasAdd) {
    preluArg = biasArg;
    biasArg = undefined;
  }
  const leakyreluAlpha =
      getParamValue('leakyreluAlpha', node, tensorMap, context) as number;

  return {
    stride,
    pad,
    dataFormat,
    dilations,
    biasArg,
    preluArg,
    activationFunc,
    leakyreluAlpha
  };
}

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext, ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'Conv1D': {
          const stride =
              getParamValue('stride', node, tensorMap, context) as number;
          const pad = getParamValue('pad', node, tensorMap, context);
          const dataFormat =
              (getParamValue('dataFormat', node, tensorMap, context) as string)
                  .toUpperCase();
          const dilation =
              getParamValue('dilation', node, tensorMap, context) as number;
          return [ops.conv1d(
              getParamValue('x', node, tensorMap, context) as Tensor3D,
              getParamValue('filter', node, tensorMap, context) as Tensor3D,
              stride, pad as 'valid' | 'same', dataFormat as 'NWC' | 'NCW',
              dilation)];
        }
        case 'Conv2D': {
          const stride =
              getParamValue('strides', node, tensorMap, context) as number[];
          const pad = getPadding(node, tensorMap, context);
          const dataFormat =
              (getParamValue('dataFormat', node, tensorMap, context) as string)
                  .toUpperCase();
          const dilations =
              getParamValue('dilations', node, tensorMap, context) as number[];
          return [ops.conv2d(
              getParamValue('x', node, tensorMap, context) as Tensor3D |
                  Tensor4D,
              getParamValue('filter', node, tensorMap, context) as Tensor4D,
              [stride[1], stride[2]], pad as 'valid' | 'same',
              dataFormat as 'NHWC' | 'NCHW', [dilations[1], dilations[2]])];
        }
        case '_FusedConv2D': {
          const {
            stride,
            pad,
            dataFormat,
            dilations,
            biasArg,
            preluArg,
            activationFunc,
            leakyreluAlpha
          } = fusedConvAndDepthWiseParams(node, tensorMap, context);

          return [ops.fused.conv2d({
            x: getParamValue('x', node, tensorMap, context) as Tensor3D |
                Tensor4D,
            filter: getParamValue('filter', node, tensorMap, context) as
                Tensor4D,
            strides: [stride[1], stride[2]],
            pad: pad as 'valid' | 'same',
            dataFormat: dataFormat as 'NHWC' | 'NCHW',
            dilations: [dilations[1], dilations[2]],
            bias: biasArg,
            activation: activationFunc as tfOps.fused.Activation,
            preluActivationWeights: preluArg,
            leakyreluAlpha
          })];
        }

        case 'FusedDepthwiseConv2dNative': {
          const {
            stride,
            pad,
            dataFormat,
            dilations,
            biasArg,
            preluArg,
            activationFunc,
            leakyreluAlpha,
          } = fusedConvAndDepthWiseParams(node, tensorMap, context);

          return [ops.fused.depthwiseConv2d({
            x: getParamValue('x', node, tensorMap, context) as Tensor3D |
                Tensor4D,
            filter: getParamValue('filter', node, tensorMap, context) as
                Tensor4D,
            strides: [stride[1], stride[2]],
            pad: pad as 'valid' | 'same',
            dataFormat: dataFormat as 'NHWC' | 'NCHW',
            dilations: [dilations[1], dilations[2]],
            bias: biasArg,
            activation: activationFunc as tfOps.fused.Activation,
            preluActivationWeights: preluArg,
            leakyreluAlpha
          })];
        }
        case 'Conv2DBackpropInput':
        case 'Conv2dTranspose': {
          const shape = getParamValue(
                            'outputShape', node, tensorMap,
                            context) as [number, number, number] |
              [number, number, number, number];
          const stride =
              getParamValue('strides', node, tensorMap, context) as number[];
          const pad = getPadding(node, tensorMap, context);
          return [ops.conv2dTranspose(
              getParamValue('x', node, tensorMap, context) as Tensor3D |
                  Tensor4D,
              getParamValue('filter', node, tensorMap, context) as Tensor4D,
              shape, [stride[1], stride[2]], pad as 'valid' | 'same')];
        }
        case 'DepthwiseConv2dNative':
        case 'DepthwiseConv2d': {
          const stride =
              getParamValue('strides', node, tensorMap, context) as number[];
          const pad = getPadding(node, tensorMap, context);
          const dilations =
              getParamValue('dilations', node, tensorMap, context) as number[];
          const dataFormat =
              (getParamValue('dataFormat', node, tensorMap, context) as string)
                  .toUpperCase();

          return [ops.depthwiseConv2d(
              getParamValue('input', node, tensorMap, context) as Tensor3D |
                  Tensor4D,
              getParamValue('filter', node, tensorMap, context) as Tensor4D,
              [stride[1], stride[2]], pad as 'valid' | 'same',
              dataFormat as 'NHWC' | 'NCHW', [dilations[1], dilations[2]])];
        }
        case 'Conv3D': {
          const stride =
              getParamValue('strides', node, tensorMap, context) as number[];
          const pad = getParamValue('pad', node, tensorMap, context);
          const dataFormat =
              (getParamValue('dataFormat', node, tensorMap, context) as string)
                  .toUpperCase();
          const dilations =
              getParamValue('dilations', node, tensorMap, context) as number[];
          return [ops.conv3d(
              getParamValue('x', node, tensorMap, context) as Tensor4D |
                  Tensor<Rank.R5>,
              getParamValue('filter', node, tensorMap, context) as
                  Tensor<Rank.R5>,
              [stride[1], stride[2], stride[3]], pad as 'valid' | 'same',
              dataFormat as 'NDHWC' | 'NCDHW',
              [dilations[1], dilations[2], dilations[3]])];
        }
        case 'AvgPool': {
          const stride =
              getParamValue('strides', node, tensorMap, context) as number[];
          const pad = getParamValue('pad', node, tensorMap, context);
          const kernelSize =
              getParamValue('kernelSize', node, tensorMap, context) as number[];

          return [ops.avgPool(
              getParamValue('x', node, tensorMap, context) as Tensor3D |
                  Tensor4D,
              [kernelSize[1], kernelSize[2]], [stride[1], stride[2]],
              pad as 'valid' | 'same')];
        }
        case 'MaxPool': {
          const stride =
              getParamValue('strides', node, tensorMap, context) as number[];
          const pad = getParamValue('pad', node, tensorMap, context);
          const kernelSize =
              getParamValue('kernelSize', node, tensorMap, context) as number[];

          return [ops.maxPool(
              getParamValue('x', node, tensorMap, context) as Tensor3D |
                  Tensor4D,
              [kernelSize[1], kernelSize[2]], [stride[1], stride[2]],
              pad as 'valid' | 'same')];
        }
        case 'MaxPoolWithArgmax': {
          const stride =
              getParamValue('strides', node, tensorMap, context) as number[];
          const pad = getParamValue('pad', node, tensorMap, context);
          const kernelSize =
              getParamValue('kernelSize', node, tensorMap, context) as number[];
          const includeBatchInIndex =
              getParamValue('includeBatchInIndex', node, tensorMap, context) as
              boolean;
          const {result, indexes} = ops.maxPoolWithArgmax(
              getParamValue('x', node, tensorMap, context) as Tensor4D,
              [kernelSize[1], kernelSize[2]], [stride[1], stride[2]],
              pad as 'valid' | 'same', includeBatchInIndex);
          return [result, indexes];
        }
        case 'AvgPool3D': {
          const stride =
              getParamValue('strides', node, tensorMap, context) as number[];
          const pad = getParamValue('pad', node, tensorMap, context);
          const kernelSize =
              getParamValue('kernelSize', node, tensorMap, context) as number[];

          return [ops.avgPool3d(
              getParamValue('x', node, tensorMap, context) as Tensor5D,
              [kernelSize[1], kernelSize[2], kernelSize[3]],
              [stride[1], stride[2], stride[3]], pad as 'valid' | 'same')];
        }

        case 'MaxPool3D': {
          const stride =
              getParamValue('strides', node, tensorMap, context) as number[];
          const pad = getParamValue('pad', node, tensorMap, context);
          const kernelSize =
              getParamValue('kernelSize', node, tensorMap, context) as number[];

          return [ops.maxPool3d(
              getParamValue('x', node, tensorMap, context) as Tensor5D,
              [kernelSize[1], kernelSize[2], kernelSize[3]],
              [stride[1], stride[2], stride[3]], pad as 'valid' | 'same')];
        }

        case 'Dilation2D': {
          const strides =
              getParamValue('strides', node, tensorMap, context) as number[];
          const pad = getParamValue('pad', node, tensorMap, context);
          const dilations =
              getParamValue('dilations', node, tensorMap, context) as number[];

          // strides: [1, stride_height, stride_width, 1].
          const strideHeight = strides[1];
          const strideWidth = strides[2];

          // dilations: [1, dilation_height, dilation_width, 1].
          const dilationHeight = dilations[1];
          const dilationWidth = dilations[2];

          return [ops.dilation2d(
              getParamValue('x', node, tensorMap, context) as Tensor3D |
                  Tensor4D,
              getParamValue('filter', node, tensorMap, context) as Tensor3D,
              [strideHeight, strideWidth], pad as 'valid' | 'same',
              [dilationHeight, dilationWidth], 'NHWC' /* dataFormat */)];
        }

        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'convolution';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
     ops = tfOps):
        Tensor[] => {
          switch (node.op) {
            case 'LowerBound': {
              const sortedSequence =
                  getParamValue('sortedSequence', node, tensorMap, context) as
                  Tensor;
              const values =
                  getParamValue('values', node, tensorMap, context) as Tensor;
              return [ops.lowerBound(sortedSequence, values)];
            }
            case 'TopKV2': {
              const x = getParamValue('x', node, tensorMap, context) as Tensor;
              const k = getParamValue('k', node, tensorMap, context) as number;
              const sorted =
                  getParamValue('sorted', node, tensorMap, context) as boolean;
              const result = ops.topk(x, k, sorted);
              return [result.values, result.indices];
            }
            case 'UpperBound': {
              const sortedSequence =
                  getParamValue('sortedSequence', node, tensorMap, context) as
                  Tensor;
              const values =
                  getParamValue('values', node, tensorMap, context) as Tensor;
              return [ops.upperBound(sortedSequence, values)];
            }
            case 'Unique': {
              const x = getParamValue('x', node, tensorMap, context) as Tensor;
              const result = ops.unique(x);
              return [result.values, result.indices];
            }
            case 'UniqueV2': {
              const x = getParamValue('x', node, tensorMap, context) as Tensor;
              const axis =
                  getParamValue('axis', node, tensorMap, context) as number;
              const result = ops.unique(x, axis);
              return [result.values, result.indices];
            }
            default:
              throw TypeError(`Node type ${node.op} is not implemented`);
          }
        };

export const CATEGORY = 'evaluation';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext, ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'ResizeBilinear': {
          const images =
              getParamValue('images', node, tensorMap, context) as Tensor;
          const size =
              getParamValue('size', node, tensorMap, context) as number[];
          const alignCorners =
              getParamValue('alignCorners', node, tensorMap, context) as
              boolean;
          const halfPixelCenters =
              getParamValue('halfPixelCenters', node, tensorMap, context) as
              boolean;
          return [ops.image.resizeBilinear(
              images as Tensor3D | Tensor4D, [size[0], size[1]], alignCorners,
              halfPixelCenters)];
        }
        case 'ResizeNearestNeighbor': {
          const images =
              getParamValue('images', node, tensorMap, context) as Tensor;
          const size =
              getParamValue('size', node, tensorMap, context) as number[];
          const alignCorners =
              getParamValue('alignCorners', node, tensorMap, context) as
              boolean;
          const halfPixelCenters =
              getParamValue('halfPixelCenters', node, tensorMap, context) as
              boolean;
          return [ops.image.resizeNearestNeighbor(
              images as Tensor3D | Tensor4D, [size[0], size[1]], alignCorners,
              halfPixelCenters)];
        }
        case 'CropAndResize': {
          const image =
              getParamValue('image', node, tensorMap, context) as Tensor;
          const boxes =
              getParamValue('boxes', node, tensorMap, context) as Tensor;
          const boxInd =
              getParamValue('boxInd', node, tensorMap, context) as Tensor;
          const cropSize =
              getParamValue('cropSize', node, tensorMap, context) as number[];
          const method =
              getParamValue('method', node, tensorMap, context) as string;
          const extrapolationValue =
              getParamValue('extrapolationValue', node, tensorMap, context) as
              number;
          return [ops.image.cropAndResize(
              image as Tensor4D, boxes as Tensor2D, boxInd as Tensor1D,
              cropSize as [number, number], method as 'bilinear' | 'nearest',
              extrapolationValue)];
        }
        case 'ImageProjectiveTransformV3': {
          const images =
              getParamValue('images', node, tensorMap, context) as Tensor;
          const transforms =
              getParamValue('transforms', node, tensorMap, context) as Tensor;
          const outputShape =
              getParamValue('outputShape', node, tensorMap, context) as
              number[];
          const fillValue =
              getParamValue('fillValue', node, tensorMap, context) as number;
          const interpolation =
              getParamValue('interpolation', node, tensorMap, context) as
              string;
          const fillMode =
              getParamValue('fillMode', node, tensorMap, context) as string;
          return [ops.image.transform(
              images as Tensor4D,
              transforms as Tensor2D,
              interpolation.toLowerCase() as 'bilinear' | 'nearest',
              fillMode.toLowerCase() as 'constant' | 'reflect' | 'wrap' | 'nearest',
              fillValue,
              outputShape as [number, number])];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'image';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor, Tensor1D, Tensor2D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext, ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'Max': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number[];
          const keepDims =
              getParamValue('keepDims', node, tensorMap, context) as boolean;
          return [ops.max(
              getParamValue('x', node, tensorMap, context) as Tensor, axis,
              keepDims)];
        }
        case 'Mean': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number[];
          const keepDims =
              getParamValue('keepDims', node, tensorMap, context) as boolean;
          return [ops.mean(
              getParamValue('x', node, tensorMap, context) as Tensor, axis,
              keepDims)];
        }
        case 'Min': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number[];
          const keepDims =
              getParamValue('keepDims', node, tensorMap, context) as boolean;
          return [ops.min(
              getParamValue('x', node, tensorMap, context) as Tensor, axis,
              keepDims)];
        }
        case 'Sum': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number[];
          const keepDims =
              getParamValue('keepDims', node, tensorMap, context) as boolean;
          return [ops.sum(
              getParamValue('x', node, tensorMap, context) as Tensor, axis,
              keepDims)];
        }
        case 'All': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number[];
          const keepDims =
              getParamValue('keepDims', node, tensorMap, context) as boolean;
          return [ops.all(
              getParamValue('x', node, tensorMap, context) as Tensor, axis,
              keepDims)];
        }
        case 'Any': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number[];
          const keepDims =
              getParamValue('keepDims', node, tensorMap, context) as boolean;
          return [ops.any(
              getParamValue('x', node, tensorMap, context) as Tensor, axis,
              keepDims)];
        }
        case 'ArgMax': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number;
          return [ops.argMax(
              getParamValue('x', node, tensorMap, context) as Tensor, axis)];
        }
        case 'ArgMin': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number;
          return [ops.argMin(
              getParamValue('x', node, tensorMap, context) as Tensor, axis)];
        }
        case 'Prod': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number[];
          const keepDims =
              getParamValue('keepDims', node, tensorMap, context) as boolean;
          return [ops.prod(
              getParamValue('x', node, tensorMap, context) as Tensor, axis,
              keepDims)];
        }
        case 'Cumprod': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number;
          const exclusive =
              getParamValue('exclusive', node, tensorMap, context) as boolean;
          const reverse =
              getParamValue('reverse', node, tensorMap, context) as boolean;
          return [ops.cumprod(
              getParamValue('x', node, tensorMap, context) as Tensor, axis,
              exclusive, reverse)];
        }
        case 'Cumsum': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number;
          const exclusive =
              getParamValue('exclusive', node, tensorMap, context) as boolean;
          const reverse =
              getParamValue('reverse', node, tensorMap, context) as boolean;
          return [ops.cumsum(
              getParamValue('x', node, tensorMap, context) as Tensor, axis,
              exclusive, reverse)];
        }
        case 'Bincount':
          const x = getParamValue('x', node, tensorMap, context) as Tensor1D;
          const weights =
              getParamValue('weights', node, tensorMap, context) as Tensor1D;
          const size =
              getParamValue('size', node, tensorMap, context) as number;

          return [ops.bincount(x, weights, size)];
        case 'DenseBincount': {
          const x = getParamValue('x', node, tensorMap, context) as Tensor1D |
              Tensor2D;
          const weights =
              getParamValue('weights', node, tensorMap, context) as Tensor1D |
              Tensor2D;
          const size =
              getParamValue('size', node, tensorMap, context) as number;

          const binaryOutput =
              getParamValue('binaryOutput', node, tensorMap, context) as
              boolean;

          return [ops.denseBincount(x, weights, size, binaryOutput)];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'reduction';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor, Tensor3D, Tensor4D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext, ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'EuclideanNorm':
          return [ops.euclideanNorm(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('axis', node, tensorMap, context) as number[],
              getParamValue('keepDims', node, tensorMap, context) as boolean)];
        case 'FusedBatchNorm':
        case 'FusedBatchNormV2': {
          return [ops.batchNorm(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('mean', node, tensorMap, context) as Tensor,
              getParamValue('variance', node, tensorMap, context) as Tensor,
              getParamValue('offset', node, tensorMap, context) as Tensor,
              getParamValue('scale', node, tensorMap, context) as Tensor,
              getParamValue('epsilon', node, tensorMap, context) as number)];
        }
        case 'FusedBatchNormV3': {
          return [ops.batchNorm(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('mean', node, tensorMap, context) as Tensor,
              getParamValue('variance', node, tensorMap, context) as Tensor,
              getParamValue('offset', node, tensorMap, context) as Tensor,
              getParamValue('scale', node, tensorMap, context) as Tensor,
              getParamValue('epsilon', node, tensorMap, context) as number)];
        }
        case 'LRN': {
          return [ops.localResponseNormalization(
              getParamValue('x', node, tensorMap, context) as Tensor3D |
                  Tensor4D,
              getParamValue('radius', node, tensorMap, context) as number,
              getParamValue('bias', node, tensorMap, context) as number,
              getParamValue('alpha', node, tensorMap, context) as number,
              getParamValue('beta', node, tensorMap, context) as number)];
        }
        case 'Softmax': {
          return [ops.softmax(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'LogSoftmax': {
          return [ops.logSoftmax(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'normalization';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {DataType, scalar, Tensor} from '@tensorflow/tfjs-core';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {TensorArray} from '../../executor/tensor_array';
import {fromTensor, reserve, scatter, split} from '../../executor/tensor_list';
import {InternalOpAsyncExecutor, Node} from '../types';

import {cloneTensor, getParamValue, getTensor} from './utils';

export const executeOp: InternalOpAsyncExecutor = async(
    node: Node, tensorMap: NamedTensorsMap,
    context: ExecutionContext): Promise<Tensor[]> => {
  switch (node.op) {
    case 'If':
    case 'StatelessIf': {
      const thenFunc =
          getParamValue('thenBranch', node, tensorMap, context) as string;
      const elseFunc =
          getParamValue('elseBranch', node, tensorMap, context) as string;
      const cond = getParamValue('cond', node, tensorMap, context) as Tensor;
      const args = getParamValue('args', node, tensorMap, context) as Tensor[];
      const condValue = await cond.data();
      if (condValue[0]) {
        return context.functionMap[thenFunc].executeFunctionAsync(
            args, context.tensorArrayMap, context.tensorListMap);
      } else {
        return context.functionMap[elseFunc].executeFunctionAsync(
            args, context.tensorArrayMap, context.tensorListMap);
      }
    }
    case 'While':
    case 'StatelessWhile': {
      const bodyFunc =
          getParamValue('body', node, tensorMap, context) as string;
      const condFunc =
          getParamValue('cond', node, tensorMap, context) as string;
      const args = getParamValue('args', node, tensorMap, context) as Tensor[];

      // Calculate the condition of the loop
      const condResult =
          (await context.functionMap[condFunc].executeFunctionAsync(
              args, context.tensorArrayMap, context.tensorListMap));
      const argIds = args.map(tensor => tensor.id);
      let condValue = await condResult[0].data();
      // Dispose the intermediate tensors for condition function
      condResult.forEach(tensor => {
        if (!tensor.kept && argIds.indexOf(tensor.id) === -1) {
          tensor.dispose();
        }
      });

      let result: Tensor[] = args;

      while (condValue[0]) {
        // Record the previous result for intermediate tensor tracking
        const origResult = result;
        // Execution the body of the loop
        result = await context.functionMap[bodyFunc].executeFunctionAsync(
            result, context.tensorArrayMap, context.tensorListMap);
        const resultIds = result.map(tensor => tensor.id);

        // Dispose the intermediate tensor for body function that is not global
        // kept, not input/output of the body function
        origResult.forEach(tensor => {
          if (!tensor.kept && argIds.indexOf(tensor.id) === -1 &&
              resultIds.indexOf(tensor.id) === -1) {
            tensor.dispose();
          }
        });

        // Recalcuate the condition of the loop using the latest results.
        const condResult =
            (await context.functionMap[condFunc].executeFunctionAsync(
                result, context.tensorArrayMap, context.tensorListMap));
        condValue = await condResult[0].data();
        // Dispose the intermediate tensors for condition function
        condResult.forEach(tensor => {
          if (!tensor.kept && argIds.indexOf(tensor.id) === -1 &&
              resultIds.indexOf(tensor.id) === -1) {
            tensor.dispose();
          }
        });
      }
      return result;
    }
    case 'LoopCond': {
      const pred = getParamValue('pred', node, tensorMap, context) as Tensor;
      return [cloneTensor(pred)];
    }
    case 'Switch': {
      const pred = getParamValue('pred', node, tensorMap, context) as Tensor;
      let data = getParamValue('data', node, tensorMap, context) as Tensor;
      if (!data.kept) {
        data = cloneTensor(data);
      }
      // Outputs nodes :0 => false, :1 => true
      return (await pred.data())[0] ? [undefined, data] : [data, undefined];
    }
    case 'Merge': {
      const inputName = node.inputNames.find(
          name => getTensor(name, tensorMap, context) !== undefined);
      if (inputName) {
        const data = getTensor(inputName, tensorMap, context);
        return [cloneTensor(data)];
      }
      return undefined;
    }
    case 'Enter': {
      const frameId =
          getParamValue('frameName', node, tensorMap, context) as string;
      const data = getParamValue('tensor', node, tensorMap, context) as Tensor;
      context.enterFrame(frameId);
      return [cloneTensor(data)];
    }
    case 'Exit': {
      const data = getParamValue('tensor', node, tensorMap, context) as Tensor;
      context.exitFrame();
      return [cloneTensor(data)];
    }
    case 'NextIteration': {
      const data = getParamValue('tensor', node, tensorMap, context) as Tensor;
      context.nextIteration();
      return [cloneTensor(data)];
    }
    case 'TensorArrayV3': {
      const size = getParamValue('size', node, tensorMap, context) as number;
      const dtype =
          getParamValue('dtype', node, tensorMap, context) as DataType;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const dynamicSize =
          getParamValue('dynamicSize', node, tensorMap, context) as boolean;
      const clearAfterRead =
          getParamValue('clearAfterRead', node, tensorMap, context) as boolean;
      const identicalElementShapes =
          getParamValue('identicalElementShapes', node, tensorMap, context) as
          boolean;
      const name = getParamValue('name', node, tensorMap, context) as string;
      const tensorArray = new TensorArray(
          name, dtype, size, elementShape, identicalElementShapes, dynamicSize,
          clearAfterRead);
      context.addTensorArray(tensorArray);
      return [tensorArray.idTensor, scalar(1.0)];
    }
    case 'TensorArrayWriteV3': {
      const id =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const index = getParamValue('index', node, tensorMap, context) as number;
      const writeTensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const writeTensorArray = context.getTensorArray(id.id);
      writeTensorArray.write(index, writeTensor);
      return [writeTensorArray.idTensor];
    }
    case 'TensorArrayReadV3': {
      const readId =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const readIndex =
          getParamValue('index', node, tensorMap, context) as number;
      const readTensorArray = context.getTensorArray(readId.id);
      return [readTensorArray.read(readIndex)];
    }
    case 'TensorArrayGatherV3': {
      const gatherId =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const gatherIndices =
          getParamValue('indices', node, tensorMap, context) as number[];
      const gatherDtype =
          getParamValue('dtype', node, tensorMap, context) as DataType;
      const gatherTensorArray = context.getTensorArray(gatherId.id);
      return [gatherTensorArray.gather(gatherIndices, gatherDtype)];
    }
    case 'TensorArrayScatterV3': {
      const scatterId =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const scatterIndices =
          getParamValue('indices', node, tensorMap, context) as number[];
      const scatterTensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const scatterTensorArray = context.getTensorArray(scatterId.id);
      scatterTensorArray.scatter(scatterIndices, scatterTensor);
      return [scatterTensorArray.idTensor];
    }
    case 'TensorArrayConcatV3': {
      const concatId =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const concatTensorArray = context.getTensorArray(concatId.id);
      const concatDtype =
          getParamValue('dtype', node, tensorMap, context) as DataType;
      return [concatTensorArray.concat(concatDtype)];
    }
    case 'TensorArraySplitV3': {
      const splitId =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const splitTensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const lengths =
          getParamValue('lengths', node, tensorMap, context) as number[];
      const splitTensorArray = context.getTensorArray(splitId.id);
      splitTensorArray.split(lengths, splitTensor);
      return [splitTensorArray.idTensor];
    }
    case 'TensorArraySizeV3': {
      const sizeId =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const sizeTensorArray = context.getTensorArray(sizeId.id);
      return [scalar(sizeTensorArray.size(), 'int32')];
    }
    case 'TensorArrayCloseV3': {
      const closeId =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const closeTensorArray = context.getTensorArray(closeId.id);
      closeTensorArray.clearAndClose();
      return [closeTensorArray.idTensor];
    }
    case 'TensorListSetItem': {
      const idTensor =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const index = getParamValue('index', node, tensorMap, context) as number;
      const writeTensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const tensorList = context.getTensorList(idTensor.id);
      tensorList.setItem(index, writeTensor);
      return [tensorList.idTensor];
    }
    case 'TensorListGetItem': {
      const idTensor =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const readIndex =
          getParamValue('index', node, tensorMap, context) as number;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];

      const elementDType =
          getParamValue('elementDType', node, tensorMap, context) as DataType;
      const tensorList = context.getTensorList(idTensor.id);
      return [tensorList.getItem(readIndex, elementShape, elementDType)];
    }
    case 'TensorListScatterV2':
    case 'TensorListScatter': {
      const scatterIndices =
          getParamValue('indices', node, tensorMap, context) as number[];
      const scatterTensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const numElements =
          getParamValue('numElements', node, tensorMap, context) as number;
      const tensorList =
          scatter(scatterTensor, scatterIndices, elementShape, numElements);
      context.addTensorList(tensorList);
      return [tensorList.idTensor];
    }
    case 'TensorListReserve':
    case 'EmptyTensorList': {
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const elementDtype =
          getParamValue('elementDType', node, tensorMap, context) as DataType;
      let numElementsParam;

      if (node.op === 'TensorListReserve') {
        numElementsParam = 'numElements';
      } else {
        numElementsParam = 'maxNumElements';
      }

      const numElements =
          getParamValue(numElementsParam, node, tensorMap, context) as number;
      const maxNumElements = node.op === 'TensorListReserve' ? -1 : numElements;
      const tensorList =
          reserve(elementShape, elementDtype, numElements, maxNumElements);
      context.addTensorList(tensorList);
      return [tensorList.idTensor];
    }
    case 'TensorListGather': {
      const gatherId =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const gatherIndices =
          getParamValue('indices', node, tensorMap, context) as number[];
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const elementDtype =
          getParamValue('elementDType', node, tensorMap, context) as DataType;
      const tensorList = context.getTensorList(gatherId.id);
      return [tensorList.gather(gatherIndices, elementDtype, elementShape)];
    }
    case 'TensorListStack': {
      const idTensor =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const elementDtype =
          getParamValue('elementDType', node, tensorMap, context) as DataType;
      const numElements =
          getParamValue('numElements', node, tensorMap, context) as number;
      const tensorList = context.getTensorList(idTensor.id);
      return [tensorList.stack(elementShape, elementDtype, numElements)];
    }
    case 'TensorListFromTensor': {
      const tensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const elementDtype =
          getParamValue('elementDType', node, tensorMap, context) as DataType;
      const tensorList = fromTensor(tensor, elementShape, elementDtype);
      context.addTensorList(tensorList);
      return [tensorList.idTensor];
    }
    case 'TensorListConcat':
    case 'TensorListConcatV2': {
      const concatId =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const tensorList = context.getTensorList(concatId.id);
      const concatDtype =
          getParamValue('dtype', node, tensorMap, context) as DataType;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      return [tensorList.concat(concatDtype, elementShape)];
    }
    case 'TensorListPushBack': {
      const idTensor =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const writeTensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const tensorList = context.getTensorList(idTensor.id);
      tensorList.pushBack(writeTensor);
      return [tensorList.idTensor];
    }
    case 'TensorListPopBack': {
      const idTensor =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const elementDType =
          getParamValue('elementDType', node, tensorMap, context) as DataType;
      const tensorList = context.getTensorList(idTensor.id);
      return [tensorList.popBack(elementShape, elementDType)];
    }
    case 'TensorListSplit': {
      const splitTensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const lengths =
          getParamValue('lengths', node, tensorMap, context) as number[];

      const tensorList = split(splitTensor, lengths, elementShape);
      context.addTensorList(tensorList);
      return [tensorList.idTensor];
    }
    case 'TensorListLength': {
      const idTensor =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const tensorList = context.getTensorList(idTensor.id);
      return [scalar(tensorList.size(), 'int32')];
    }
    case 'TensorListResize': {
      const idTensor =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const size = getParamValue('size', node, tensorMap, context) as number;

      const srcTensorList = context.getTensorList(idTensor.id);
      const destTensorList = srcTensorList.resize(size);
      context.addTensorList(destTensorList);
      return [destTensorList.idTensor];
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'control';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor, Tensor4D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
     ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'Cast': {
          return [ops.cast(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('dtype', node, tensorMap, context) as 'int32' |
                  'float32' | 'bool')];
        }
        case 'ExpandDims': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number;
          return [ops.expandDims(
              getParamValue('x', node, tensorMap, context) as Tensor, axis)];
        }
        case 'Squeeze': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number[];
          return [ops.squeeze(
              getParamValue('x', node, tensorMap, context) as Tensor, axis)];
        }

        case 'Reshape': {
          return [ops.reshape(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('shape', node, tensorMap, context) as number[])];
        }
        case 'EnsureShape': {
          return [ops.ensureShape(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('shape', node, tensorMap, context) as number[])];
        }
        case 'MirrorPad': {
          return [ops.mirrorPad(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('padding', node, tensorMap, context) as
                  Array<[number, number]>,
              getParamValue('mode', node, tensorMap, context) as 'reflect' |
                  'symmetric')];
        }
        case 'PadV2':
        case 'Pad': {
          return [ops.pad(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('padding', node, tensorMap, context) as
                  Array<[number, number]>,
              getParamValue('constantValue', node, tensorMap, context) as
                  number)];
        }
        case 'SpaceToBatchND': {
          const blockShape =
              getParamValue('blockShape', node, tensorMap, context) as number[];
          const paddings =
              getParamValue('paddings', node, tensorMap, context) as number[][];
          return [ops.spaceToBatchND(
              getParamValue('x', node, tensorMap, context) as Tensor,
              blockShape, paddings)];
        }
        case 'BatchToSpaceND': {
          const blockShape =
              getParamValue('blockShape', node, tensorMap, context) as number[];
          const crops =
              getParamValue('crops', node, tensorMap, context) as number[][];
          return [ops.batchToSpaceND(
              getParamValue('x', node, tensorMap, context) as Tensor,
              blockShape, crops)];
        }
        case 'DepthToSpace': {
          const blockSize =
              getParamValue('blockSize', node, tensorMap, context) as number;
          const dataFormat =
              (getParamValue('dataFormat', node, tensorMap, context) as
               string).toUpperCase() as 'NHWC' |
              'NCHW';
          return [ops.depthToSpace(
              getParamValue('x', node, tensorMap, context) as Tensor4D,
              blockSize, dataFormat)];
        }
        case 'BroadcastTo': {
          return [ops.broadcastTo(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('shape', node, tensorMap, context) as number[])];
        }
        case 'BroadcastArgs': {
          return [ops.broadcastArgs(
              getParamValue('s0', node, tensorMap, context) as Tensor,
              getParamValue('s1', node, tensorMap, context) as Tensor)];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'transformation';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Scalar, Tensor, Tensor1D, tidy, util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
     ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'ConcatV2':
        case 'Concat': {
          const n = getParamValue('n', node, tensorMap, context) as number;
          const axis =
              getParamValue('axis', node, tensorMap, context) as number;
          let inputs =
              getParamValue('tensors', node, tensorMap, context) as Tensor[];
          inputs = inputs.slice(0, n);
          return [ops.concat(inputs, axis)];
        }
        case 'Gather': {
          const input = getParamValue('x', node, tensorMap, context) as Tensor;
          const indices =
              getParamValue('indices', node, tensorMap, context) as Tensor1D;
          return [ops.gather(input, ops.cast(indices, 'int32'), 0)];
        }
        case 'GatherV2': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number;
          const batchDims =
              getParamValue('batchDims', node, tensorMap, context) as number;
          const input = getParamValue('x', node, tensorMap, context) as Tensor;
          const indices =
              getParamValue('indices', node, tensorMap, context) as Tensor1D;
          return [ops.gather(
              input, ops.cast(indices, 'int32'), axis, batchDims)];
        }
        case 'Reverse': {
          const dims =
              getParamValue('dims', node, tensorMap, context) as boolean[];
          const axis = [];
          for (let i = 0; i < dims.length; i++) {
            if (dims[i]) {
              axis.push(i);
            }
          }
          const input = getParamValue('x', node, tensorMap, context) as Tensor;
          return [ops.reverse(input, axis)];
        }
        case 'ReverseV2': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number[];
          const input = getParamValue('x', node, tensorMap, context) as Tensor;
          return [ops.reverse(input, axis)];
        }
        case 'Slice': {
          // tslint:disable-next-line:no-any
          const begin = getParamValue('begin', node, tensorMap, context) as any;
          // tslint:disable-next-line:no-any
          const size = getParamValue('size', node, tensorMap, context) as any;
          return [ops.slice(
              getParamValue('x', node, tensorMap, context) as Tensor, begin,
              size)];
        }
        case 'StridedSlice': {
          const begin =
              getParamValue('begin', node, tensorMap, context) as number[];
          const end =
              getParamValue('end', node, tensorMap, context) as number[];
          const strides =
              getParamValue('strides', node, tensorMap, context) as number[];
          const beginMask =
              getParamValue('beginMask', node, tensorMap, context) as number;
          const endMask =
              getParamValue('endMask', node, tensorMap, context) as number;
          const ellipsisMask =
              getParamValue('ellipsisMask', node, tensorMap, context) as number;
          const newAxisMask =
              getParamValue('newAxisMask', node, tensorMap, context) as number;
          const shrinkAxisMask =
              getParamValue('shrinkAxisMask', node, tensorMap, context) as
              number;
          const tensor = getParamValue('x', node, tensorMap, context) as Tensor;

          return [ops.stridedSlice(
              tensor, begin, end, strides, beginMask, endMask, ellipsisMask,
              newAxisMask, shrinkAxisMask)];
        }
        case 'Pack': {
          return tidy(() => {
            const axis =
                getParamValue('axis', node, tensorMap, context) as number;
            const tensors =
                getParamValue('tensors', node, tensorMap, context) as Tensor[];
            // Reshape the tensors to the first tensor's shape if they don't
            // match.
            const shape = tensors[0].shape;
            const squeezedShape = ops.squeeze(tensors[0]).shape;
            const mapped = tensors.map(tensor => {
              const sameShape = util.arraysEqual(tensor.shape, shape);
              if (!sameShape &&
                  !util.arraysEqual(ops.squeeze(tensor).shape, squeezedShape)) {
                throw new Error('the input tensors shape does not match');
              }
              return sameShape ? tensor : ops.reshape(tensor, shape);
            });
            return [ops.stack(mapped, axis)];
          });
        }
        case 'Unpack': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number;
          const tensor =
              getParamValue('tensor', node, tensorMap, context) as Tensor;
          return ops.unstack(tensor, axis);
        }
        case 'Tile': {
          const reps =
              getParamValue('reps', node, tensorMap, context) as number[];
          return [ops.tile(
              getParamValue('x', node, tensorMap, context) as Tensor, reps)];
        }
        case 'Split':
        case 'SplitV': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number;
          const numOrSizeSplits =
              getParamValue('numOrSizeSplits', node, tensorMap, context) as
                  number |
              number[];
          const tensor = getParamValue('x', node, tensorMap, context) as Tensor;

          return ops.split(tensor, numOrSizeSplits, axis);
        }
        case 'ScatterNd': {
          const indices =
              getParamValue('indices', node, tensorMap, context) as Tensor;
          const values =
              getParamValue('values', node, tensorMap, context) as Tensor;
          const shape =
              getParamValue('shape', node, tensorMap, context) as number[];
          return [ops.scatterND(indices, values, shape)];
        }
        case 'GatherNd': {
          const x = getParamValue('x', node, tensorMap, context) as Tensor;
          const indices =
              getParamValue('indices', node, tensorMap, context) as Tensor;
          return [ops.gatherND(x, indices)];
        }
        case 'SparseToDense': {
          const indices =
              getParamValue('sparseIndices', node, tensorMap, context) as
              Tensor;
          const shape =
              getParamValue('outputShape', node, tensorMap, context) as
              number[];
          const sparseValues =
              getParamValue('sparseValues', node, tensorMap, context) as Tensor;
          const defaultValue =
              getParamValue('defaultValue', node, tensorMap, context) as Scalar;
          return [ops.sparseToDense(
              indices, sparseValues, shape,
              sparseValues.dtype === defaultValue.dtype ?
                  defaultValue :
                  ops.cast(defaultValue, sparseValues.dtype))];
        }
        case 'TensorScatterUpdate': {
          const indices =
              getParamValue('indices', node, tensorMap, context) as Tensor;
          const values =
              getParamValue('values', node, tensorMap, context) as Tensor;
          const tensor =
              getParamValue('tensor', node, tensorMap, context) as Tensor;
          return [ops.tensorScatterUpdate(tensor, indices, values)];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'slice_join';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import {Tensor} from '@tensorflow/tfjs-core';
// tslint:disable-next-line:no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import * as tensorflow from '../data/compiled_api';
import {NamedTensorsMap} from '../data/types';
import {ExecutionContext} from '../executor/execution_context';
import {ResourceManager} from '../executor/resource_manager';

export type ParamType = 'number'|'string'|'string[]'|'number[]'|'bool'|'bool[]'|
    'shape'|'shape[]'|'tensor'|'tensors'|'dtype'|'dtype[]'|'func';
export type Category = 'arithmetic'|'basic_math'|'control'|'convolution'|
    'creation'|'custom'|'dynamic'|'evaluation'|'graph'|'hash_table'|'image'|
    'logical'|'matrices'|'normalization'|'ragged'|'reduction'|'slice_join'|
    'sparse'|'spectral'|'string'|'transformation';

// For mapping input or attributes of NodeDef into TensorFlow.js op param.
export declare interface ParamMapper {
  // tensorflow.js name for the field, it should be in camelcase format.
  name: string;
  type: ParamType;
  defaultValue?: ValueType;
  notSupported?: boolean;
}

// For mapping the input of TensorFlow NodeDef into TensorFlow.js Op param.
export declare interface InputParamMapper extends ParamMapper {
  // The first number is the starting index of the param, the second number is
  // the length of the param. If the length value is positive number, it
  // represents the true length of the param. Otherwise, it represents a
  // variable length, the value is the index go backward from the end of the
  // array.
  // For example `[0, 5]`: this param is the array of input tensors starting at
  // index 0 and with the length of 5.
  // For example `[1, -1]`: this param is the array of input tensors starting at
  // index 1 and with the `inputs.length - 1`.
  // Zero-based index at where in the input array this param starts.
  // A negative index can be used, indicating an offset from the end of the
  // sequence. slice(-2) extracts the last two elements in the sequence.
  start: number;
  // Zero-based index before where in the input array the param ends. The
  // mapping is up to but not including end. For example, start = 1, end = 4
  // includes the second element through the fourth element (elements indexed 1,
  // 2, and 3). A negative index can be used, indicating an offset from the end
  // of the sequence. start = 2, end = -1 includes the third element through the
  // second-to-last element in the sequence. If end is omitted, end is set to
  // start + 1, the mapping only include the single element at start index. If
  // end is set to 0, the mapping is through the end of the input array
  // (arr.length). If end is greater than the length of the inputs, mapping
  // inncludes through to the end of the sequence (arr.length).
  end?: number;
}

// For mapping the attributes of TensorFlow NodeDef into TensorFlow.js op param.
export declare interface AttrParamMapper extends ParamMapper {
  // TensorFlow attribute name, this should be set if the tensorflow attribute
  // name is different form the tensorflow.js name.
  tfName?: string;
  // TensorFlow deprecated attribute name, this is used to support old models.
  tfDeprecatedName?: string;
}

export interface InternalOpExecutor {
  (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
   ops?: typeof tfOps): Tensor|Tensor[];
}

export interface InternalOpAsyncExecutor {
  (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
   resourceManager?: ResourceManager, ops?: typeof tfOps): Promise<Tensor[]>;
}

export declare interface OpMapper {
  tfOpName: string;
  category?: Category;
  inputs?: InputParamMapper[];
  attrs?: AttrParamMapper[];
  outputs?: string[];
  customExecutor?: OpExecutor;
}

export declare interface Node {
  signatureKey?: string;
  name: string;
  op: string;
  category: Category;
  inputNames: string[];
  inputs: Node[];
  inputParams: {[key: string]: InputParamValue};
  attrParams: {[key: string]: ParamValue};
  children: Node[];
  rawAttrs?: {[k: string]: tensorflow.IAttrValue};
  defaultOutput?: number;
  outputs?: string[];
}

export declare interface Graph {
  nodes: {[key: string]: Node};
  placeholders: Node[];
  inputs: Node[];
  outputs: Node[];
  weights: Node[];
  signature?: tensorflow.ISignatureDef;
  functions?: {[key: string]: Graph};
  initNodes?: Node[];
}

export type ValueType = string|string[]|number|number[]|number[][]|boolean|
    boolean[]|Tensor|Tensor[];
export declare interface ParamValue {
  value?: ValueType;
  type: ParamType;
}

export declare interface InputParamValue extends ParamValue {
  inputIndexStart?: number;
  inputIndexEnd?: number;
}

export interface OpExecutor {
  (node: GraphNode): Tensor|Tensor[]|Promise<Tensor|Tensor[]>;
}

export interface GraphNode {
  inputs: Tensor[];
  attrs: {[key: string]: ValueType};
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';

import {NamedTensorsMap} from '../data/types';
import {ExecutionContext} from '../executor/execution_context';
import {ResourceManager} from '../executor/resource_manager';

import {NodeValueImpl} from './custom_op/node_value_impl';
import {getRegisteredOp} from './custom_op/register';
import * as arithmetic from './executors/arithmetic_executor';
import * as basicMath from './executors/basic_math_executor';
import * as control from './executors/control_executor';
import * as convolution from './executors/convolution_executor';
import * as creation from './executors/creation_executor';
import * as dynamic from './executors/dynamic_executor';
import * as evaluation from './executors/evaluation_executor';
import * as graph from './executors/graph_executor';
import * as hashTable from './executors/hash_table_executor';
import * as image from './executors/image_executor';
import * as logical from './executors/logical_executor';
import * as matrices from './executors/matrices_executor';
import * as normalization from './executors/normalization_executor';
import * as ragged from './executors/ragged_executor';
import * as reduction from './executors/reduction_executor';
import * as sliceJoin from './executors/slice_join_executor';
import * as sparse from './executors/sparse_executor';
import * as spectral from './executors/spectral_executor';
import * as string from './executors/string_executor';
import * as transformation from './executors/transformation_executor';
import {Node} from './types';

/**
 * Executes the op defined by the node object.
 * @param node
 * @param tensorMap contains tensors for executed nodes and weights
 * @param context contains tensors and information for running the current node.
 * @param resourceManager Optional. Contains global resources of the model.
 */
export function executeOp(
    node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
    resourceManager?: ResourceManager, tidy = tfc.tidy): tfc.Tensor[]|
    Promise<tfc.Tensor[]> {
  const value =
      ((node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext) => {
        switch (node.category) {
          case 'arithmetic':
            return tidy(() => arithmetic.executeOp(node, tensorMap, context));
          case 'basic_math':
            return tidy(() => basicMath.executeOp(node, tensorMap, context));
          case 'control':
            return control.executeOp(node, tensorMap, context);
          case 'convolution':
            return tidy(() => convolution.executeOp(node, tensorMap, context));
          case 'creation':
            return tidy(() => creation.executeOp(node, tensorMap, context));
          case 'dynamic':
            return dynamic.executeOp(node, tensorMap, context);
          case 'evaluation':
            return tidy(() => evaluation.executeOp(node, tensorMap, context));
          case 'image':
            return tidy(() => image.executeOp(node, tensorMap, context));
          case 'graph':
            return tidy(() => graph.executeOp(node, tensorMap, context));
          case 'logical':
            return tidy(() => logical.executeOp(node, tensorMap, context));
          case 'matrices':
            return tidy(() => matrices.executeOp(node, tensorMap, context));
          case 'normalization':
            return tidy(
                () => normalization.executeOp(node, tensorMap, context));
          case 'ragged':
            return tidy(() => ragged.executeOp(node, tensorMap, context));
          case 'reduction':
            return tidy(() => reduction.executeOp(node, tensorMap, context));
          case 'slice_join':
            return tidy(() => sliceJoin.executeOp(node, tensorMap, context));
          case 'sparse':
            return tidy(() => sparse.executeOp(node, tensorMap, context));
          case 'spectral':
            return tidy(() => spectral.executeOp(node, tensorMap, context));
          case 'string':
            return tidy(() => string.executeOp(node, tensorMap, context));
          case 'transformation':
            return tidy(
                () => transformation.executeOp(node, tensorMap, context));
          case 'hash_table':
            return hashTable.executeOp(
                node, tensorMap, context, resourceManager);
          case 'custom':
            const opMapper = getRegisteredOp(node.op);
            if (opMapper && opMapper.customExecutor) {
              return opMapper.customExecutor(
                  new NodeValueImpl(node, tensorMap, context));
            } else {
              throw TypeError(`Custom op ${node.op} is not registered.`);
            }
          default:
            throw TypeError(
                `Unknown op '${node.op}'. File an issue at ` +
                `https://github.com/tensorflow/tfjs/issues so we can add it` +
                `, or register a custom execution with tf.registerOp()`);
        }
      })(node, tensorMap, context);
  if (tfc.util.isPromise(value)) {
    return value.then((data) => [].concat(data));
  }
  return [].concat(value);
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

export const json = {
  '$schema': 'http://json-schema.org/draft-07/schema#',
  'definitions': {
    'OpMapper': {
      'type': 'object',
      'properties': {
        'tfOpName': {'type': 'string'},
        'category': {'$ref': '#/definitions/Category'},
        'inputs': {
          'type': 'array',
          'items': {'$ref': '#/definitions/InputParamMapper'}
        },
        'attrs': {
          'type': 'array',
          'items': {'$ref': '#/definitions/AttrParamMapper'}
        },
        'customExecutor': {'$ref': '#/definitions/OpExecutor'},
        'outputs': {'type': 'array'}
      },
      'required': ['tfOpName'],
      'additionalProperties': false
    },
    'Category': {
      'type': 'string',
      'enum': [
        'arithmetic',    'basic_math',     'control',    'convolution',
        'custom',        'dynamic',        'evaluation', 'image',
        'creation',      'graph',          'logical',    'matrices',
        'normalization', 'ragged',         'reduction',  'slice_join',
        'spectral',      'transformation', 'sparse',     'string'
      ]
    },
    'InputParamMapper': {
      'type': 'object',
      'properties': {
        'name': {'type': 'string'},
        'type': {'$ref': '#/definitions/ParamTypes'},
        'defaultValue': {
          'anyOf': [
            {'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}},
            {'type': 'number'}, {'type': 'array', 'items': {'type': 'number'}},
            {'type': 'boolean'}, {'type': 'array', 'items': {'type': 'boolean'}}
          ]
        },
        'notSupported': {'type': 'boolean'},
        'start': {'type': 'number'},
        'end': {'type': 'number'}
      },
      'required': ['name', 'start', 'type'],
      'additionalProperties': false
    },
    'ParamTypes': {
      'type': 'string',
      'enum': [
        'number', 'string', 'number[]', 'bool', 'shape', 'tensor', 'tensors',
        'dtype', 'string[]', 'func', 'dtype[]', 'bool[]'
      ]
    },
    'AttrParamMapper': {
      'type': 'object',
      'properties': {
        'name': {'type': 'string'},
        'type': {'$ref': '#/definitions/ParamTypes'},
        'defaultValue': {
          'anyOf': [
            {'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}},
            {'type': 'number'}, {'type': 'array', 'items': {'type': 'number'}},
            {'type': 'boolean'}, {'type': 'array', 'items': {'type': 'boolean'}}
          ]
        },
        'notSupported': {'type': 'boolean'},
        'tfName': {'type': 'string'},
        'tfDeprecatedName': {'type': 'string'}
      },
      'required': ['name', 'tfName', 'type'],
      'additionalProperties': false
    },
    'OpExecutor': {'type': 'object', 'additionalProperties': false}
  },
  'items': {'$ref': '#/definitions/OpMapper'},
  'type': 'array'
};

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {env} from '@tensorflow/tfjs-core';

const ENV = env();

/** Whether to keep intermediate tensors. */
ENV.registerFlag('KEEP_INTERMEDIATE_TENSORS', () => false, debugValue => {
  if (debugValue) {
    console.warn(
        'Keep intermediate tensors is ON. This will print the values of all ' +
        'intermediate tensors during model inference. Not all models ' +
        'support this mode. For details, check e2e/benchmarks/ ' +
        'model_config.js. This significantly impacts performance.');
  }
});

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import {DataType, Tensor} from '@tensorflow/tfjs-core';

import {HashTable} from '../executor/hash_table';
import {TensorArray} from '../executor/tensor_array';
import {TensorList} from '../executor/tensor_list';

export type NamedTensorMap = {
  [key: string]: Tensor
};

export type NamedTensorsMap = {
  [key: string]: Tensor[]
};

export type TensorArrayMap = {
  [key: number]: TensorArray
};

export type TensorListMap = {
  [key: number]: TensorList
};

export type HashTableMap = {
  [key: number]: HashTable
};

export interface TensorInfo {
  name: string;
  shape?: number[];
  dtype: DataType;
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */

/* tslint:disable */

/** Properties of an Any. */
export declare interface IAny {
  /** Any typeUrl */
  typeUrl?: (string|null);

  /** Any value */
  value?: (Uint8Array|null);
}

/** DataType enum. */
export enum DataType {
  // These properties must be quoted since they are used by parseDtypeParam
  // in tfjs-converter/src/operations/operation_mapper.ts to look up dtypes
  // by string name. If they are not quoted, Closure will mangle their names.

  // Not a legal value for DataType.  Used to indicate a DataType field
  // has not been set.
  'DT_INVALID' = 0,

  // Data types that all computation devices are expected to be
  // capable to support.
  'DT_FLOAT' = 1,
  'DT_DOUBLE' = 2,
  'DT_INT32' = 3,
  'DT_UINT8' = 4,
  'DT_INT16' = 5,
  'DT_INT8' = 6,
  'DT_STRING' = 7,
  'DT_COMPLEX64' = 8,  // Single-precision complex
  'DT_INT64' = 9,
  'DT_BOOL' = 10,
  'DT_QINT8' = 11,     // Quantized int8
  'DT_QUINT8' = 12,    // Quantized uint8
  'DT_QINT32' = 13,    // Quantized int32
  'DT_BFLOAT16' = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
  'DT_QINT16' = 15,    // Quantized int16
  'DT_QUINT16' = 16,   // Quantized uint16
  'DT_UINT16' = 17,
  'DT_COMPLEX128' = 18,  // Double-precision complex
  'DT_HALF' = 19,
  'DT_RESOURCE' = 20,
  'DT_VARIANT' = 21,  // Arbitrary C++ data types
  'DT_UINT32' = 22,
  'DT_UINT64' = 23,

  // Do not use!  These are only for parameters.  Every enum above
  // should have a corresponding value below (verified by types_test).
  'DT_FLOAT_REF' = 101,
  'DT_DOUBLE_REF' = 102,
  'DT_INT32_REF' = 103,
  'DT_UINT8_REF' = 104,
  'DT_INT16_REF' = 105,
  'DT_INT8_REF' = 106,
  'DT_STRING_REF' = 107,
  'DT_COMPLEX64_REF' = 108,
  'DT_INT64_REF' = 109,
  'DT_BOOL_REF' = 110,
  'DT_QINT8_REF' = 111,
  'DT_QUINT8_REF' = 112,
  'DT_QINT32_REF' = 113,
  'DT_BFLOAT16_REF' = 114,
  'DT_QINT16_REF' = 115,
  'DT_QUINT16_REF' = 116,
  'DT_UINT16_REF' = 117,
  'DT_COMPLEX128_REF' = 118,
  'DT_HALF_REF' = 119,
  'DT_RESOURCE_REF' = 120,
  'DT_VARIANT_REF' = 121,
  'DT_UINT32_REF' = 122,
  'DT_UINT64_REF' = 123,
}

/** Properties of a TensorShape. */
export declare interface ITensorShape {
  /** TensorShape dim */
  dim?: (TensorShape.IDim[]|null);

  /** TensorShape unknownRank */
  unknownRank?: (boolean|null);
}

export namespace TensorShape {
  /** Properties of a Dim. */
  export declare interface IDim {
    /** Dim size */
    size?: (number|string|null);

    /** Dim name */
    name?: (string|null);
  }
}

/** Properties of a Tensor. */
export declare interface ITensor {
  /** Tensor dtype */
  dtype?: (DataType|null);

  /** Tensor tensorShape */
  tensorShape?: (ITensorShape|null);

  /** Tensor versionNumber */
  versionNumber?: (number|null);

  /** Tensor tensorContent */
  tensorContent?: (Uint8Array|null);

  /** Tensor floatVal */
  floatVal?: (number[]|null);

  /** Tensor doubleVal */
  doubleVal?: (number[]|null);

  /** Tensor intVal */
  intVal?: (number[]|null);

  /** Tensor stringVal */
  stringVal?: (Uint8Array[]|null);

  /** Tensor scomplexVal */
  scomplexVal?: (number[]|null);

  /** Tensor int64Val */
  int64Val?: ((number | string)[]|null);

  /** Tensor boolVal */
  boolVal?: (boolean[]|null);

  /** Tensor uint32Val */
  uint32Val?: (number[]|null);

  /** Tensor uint64Val */
  uint64Val?: ((number | string)[]|null);
}

/** Properties of an AttrValue. */
export declare interface IAttrValue {
  /** AttrValue list */
  list?: (AttrValue.IListValue|null);

  /** AttrValue s */
  s?: (string|null);

  /** AttrValue i */
  i?: (number|string|null);

  /** AttrValue f */
  f?: (number|null);

  /** AttrValue b */
  b?: (boolean|null);

  /** AttrValue type */
  type?: (DataType|null);

  /** AttrValue shape */
  shape?: (ITensorShape|null);

  /** AttrValue tensor */
  tensor?: (ITensor|null);

  /** AttrValue placeholder */
  placeholder?: (string|null);

  /** AttrValue func */
  func?: (INameAttrList|null);
}

export namespace AttrValue {
  /** Properties of a ListValue. */
  export declare interface IListValue {
    /** ListValue s */
    s?: (string[]|null);

    /** ListValue i */
    i?: ((number | string)[]|null);

    /** ListValue f */
    f?: (number[]|null);

    /** ListValue b */
    b?: (boolean[]|null);

    /** ListValue type */
    type?: (DataType[]|null);

    /** ListValue shape */
    shape?: (ITensorShape[]|null);

    /** ListValue tensor */
    tensor?: (ITensor[]|null);

    /** ListValue func */
    func?: (INameAttrList[]|null);
  }
}

/** Properties of a NameAttrList. */
export declare interface INameAttrList {
  /** NameAttrList name */
  name?: (string|null);

  /** NameAttrList attr */
  attr?: ({[k: string]: IAttrValue}|null);
}

/** Properties of a NodeDef. */
export declare interface INodeDef {
  /** NodeDef name */
  name?: (string|null);

  /** NodeDef op */
  op?: (string|null);

  /** NodeDef input */
  input?: (string[]|null);

  /** NodeDef device */
  device?: (string|null);

  /** NodeDef attr */
  attr?: ({[k: string]: IAttrValue}|null);
}

/** Properties of a VersionDef. */
export declare interface IVersionDef {
  /** VersionDef producer */
  producer?: (number|null);

  /** VersionDef minConsumer */
  minConsumer?: (number|null);

  /** VersionDef badConsumers */
  badConsumers?: (number[]|null);
}

/** Properties of a GraphDef. */
export declare interface IGraphDef {
  /** GraphDef node */
  node?: (INodeDef[]|null);

  /** GraphDef versions */
  versions?: (IVersionDef|null);

  /** GraphDef library */
  library?: (IFunctionDefLibrary|null);
}

/** Properties of a CollectionDef. */
export declare interface ICollectionDef {
  /** CollectionDef nodeList */
  nodeList?: (CollectionDef.INodeList|null);

  /** CollectionDef bytesList */
  bytesList?: (CollectionDef.IBytesList|null);

  /** CollectionDef int64List */
  int64List?: (CollectionDef.IInt64List|null);

  /** CollectionDef floatList */
  floatList?: (CollectionDef.IFloatList|null);

  /** CollectionDef anyList */
  anyList?: (CollectionDef.IAnyList|null);
}

export namespace CollectionDef {
  /** Properties of a NodeList. */
  export declare interface INodeList {
    /** NodeList value */
    value?: (string[]|null);
  }

  /** Properties of a BytesList. */
  export declare interface IBytesList {
    /** BytesList value */
    value?: (Uint8Array[]|null);
  }

  /** Properties of an Int64List. */
  export declare interface IInt64List {
    /** Int64List value */
    value?: ((number | string)[]|null);
  }

  /** Properties of a FloatList. */
  export declare interface IFloatList {
    /** FloatList value */
    value?: (number[]|null);
  }

  /** Properties of an AnyList. */
  export declare interface IAnyList {
    /** AnyList value */
    value?: (IAny[]|null);
  }
}

/** Properties of a SaverDef. */
export declare interface ISaverDef {
  /** SaverDef filenameTensorName */
  filenameTensorName?: (string|null);

  /** SaverDef saveTensorName */
  saveTensorName?: (string|null);

  /** SaverDef restoreOpName */
  restoreOpName?: (string|null);

  /** SaverDef maxToKeep */
  maxToKeep?: (number|null);

  /** SaverDef sharded */
  sharded?: (boolean|null);

  /** SaverDef keepCheckpointEveryNHours */
  keepCheckpointEveryNHours?: (number|null);

  /** SaverDef version */
  version?: (SaverDef.CheckpointFormatVersion|null);
}

export namespace SaverDef {
  /** CheckpointFormatVersion enum. */
  export enum CheckpointFormatVersion {'LEGACY' = 0, 'V1' = 1, 'V2' = 2}
}

/** Properties of a TensorInfo. */
export declare interface ITensorInfo {
  /** TensorInfo name */
  name?: (string|null);

  /** TensorInfo cooSparse */
  cooSparse?: (TensorInfo.ICooSparse|null);

  /** TensorInfo dtype */
  dtype?: (DataType|string|null);

  /** TensorInfo tensorShape */
  tensorShape?: (ITensorShape|null);

  /** Resource id tensor was originally assigned to.  */
  resourceId?: (number|null);
}

export namespace TensorInfo {
  /** Properties of a CooSparse. */
  export declare interface ICooSparse {
    /** CooSparse valuesTensorName */
    valuesTensorName?: (string|null);

    /** CooSparse indicesTensorName */
    indicesTensorName?: (string|null);

    /** CooSparse denseShapeTensorName */
    denseShapeTensorName?: (string|null);
  }
}

/** Properties of a SignatureDef. */
export declare interface ISignatureDef {
  /** SignatureDef inputs */
  inputs?: ({[k: string]: ITensorInfo}|null);

  /** SignatureDef outputs */
  outputs?: ({[k: string]: ITensorInfo}|null);

  /** SignatureDef methodName */
  methodName?: (string|null);
}

/** Properties of an AssetFileDef. */
export declare interface IAssetFileDef {
  /** AssetFileDef tensorInfo */
  tensorInfo?: (ITensorInfo|null);

  /** AssetFileDef filename */
  filename?: (string|null);
}

/** Properties of an OpDef. */
export declare interface IOpDef {
  /** OpDef name */
  name?: (string|null);

  /** OpDef inputArg */
  inputArg?: (OpDef.IArgDef[]|null);

  /** OpDef outputArg */
  outputArg?: (OpDef.IArgDef[]|null);

  /** OpDef attr */
  attr?: (OpDef.IAttrDef[]|null);

  /** OpDef deprecation */
  deprecation?: (OpDef.IOpDeprecation|null);

  /** OpDef summary */
  summary?: (string|null);

  /** OpDef description */
  description?: (string|null);

  /** OpDef isCommutative */
  isCommutative?: (boolean|null);

  /** OpDef isAggregate */
  isAggregate?: (boolean|null);

  /** OpDef isStateful */
  isStateful?: (boolean|null);

  /** OpDef allowsUninitializedInput */
  allowsUninitializedInput?: (boolean|null);
}

export namespace OpDef {
  /** Properties of an ArgDef. */
  export declare interface IArgDef {
    /** ArgDef name */
    name?: (string|null);

    /** ArgDef description */
    description?: (string|null);

    /** ArgDef type */
    type?: (DataType|null);

    /** ArgDef typeAttr */
    typeAttr?: (string|null);

    /** ArgDef numberAttr */
    numberAttr?: (string|null);

    /** ArgDef typeListAttr */
    typeListAttr?: (string|null);

    /** ArgDef isRef */
    isRef?: (boolean|null);
  }

  /** Properties of an AttrDef. */
  export declare interface IAttrDef {
    /** AttrDef name */
    name?: (string|null);

    /** AttrDef type */
    type?: (string|null);

    /** AttrDef defaultValue */
    defaultValue?: (IAttrValue|null);

    /** AttrDef description */
    description?: (string|null);

    /** AttrDef hasMinimum */
    hasMinimum?: (boolean|null);

    /** AttrDef minimum */
    minimum?: (number|string|null);

    /** AttrDef allowedValues */
    allowedValues?: (IAttrValue|null);
  }

  /** Properties of an OpDeprecation. */
  export declare interface IOpDeprecation {
    /** OpDeprecation version */
    version?: (number|null);

    /** OpDeprecation explanation */
    explanation?: (string|null);
  }
}

/** Properties of an OpList. */
export declare interface IOpList {
  /** OpList op */
  op?: (IOpDef[]|null);
}

/** Properties of a MetaGraphDef. */
export declare interface IMetaGraphDef {
  /** MetaGraphDef metaInfoDef */
  metaInfoDef?: (MetaGraphDef.IMetaInfoDef|null);

  /** MetaGraphDef graphDef */
  graphDef?: (IGraphDef|null);

  /** MetaGraphDef saverDef */
  saverDef?: (ISaverDef|null);

  /** MetaGraphDef collectionDef */
  collectionDef?: ({[k: string]: ICollectionDef}|null);

  /** MetaGraphDef signatureDef */
  signatureDef?: ({[k: string]: ISignatureDef}|null);

  /** MetaGraphDef assetFileDef */
  assetFileDef?: (IAssetFileDef[]|null);
}

export namespace MetaGraphDef {
  /** Properties of a MetaInfoDef. */
  export declare interface IMetaInfoDef {
    /** MetaInfoDef metaGraphVersion */
    metaGraphVersion?: (string|null);

    /** MetaInfoDef strippedOpList */
    strippedOpList?: (IOpList|null);

    /** MetaInfoDef anyInfo */
    anyInfo?: (IAny|null);

    /** MetaInfoDef tags */
    tags?: (string[]|null);

    /** MetaInfoDef tensorflowVersion */
    tensorflowVersion?: (string|null);

    /** MetaInfoDef tensorflowGitVersion */
    tensorflowGitVersion?: (string|null);
  }
}

/** Properties of a SavedModel. */
export declare interface ISavedModel {
  /** SavedModel savedModelSchemaVersion */
  savedModelSchemaVersion?: (number|string|null);

  /** SavedModel metaGraphs */
  metaGraphs?: (IMetaGraphDef[]|null);
}

/** Properties of a FunctionDefLibrary. */
export declare interface IFunctionDefLibrary {
  /** FunctionDefLibrary function */
  'function'?: (IFunctionDef[]|null);

  /** FunctionDefLibrary gradient */
  gradient?: (IGradientDef[]|null);
}

/** Properties of a FunctionDef. */
export declare interface IFunctionDef {
  /** FunctionDef signature */
  signature?: (IOpDef|null);

  /** FunctionDef attr */
  attr?: ({[k: string]: IAttrValue}|null);

  /** FunctionDef nodeDef */
  nodeDef?: (INodeDef[]|null);

  /** FunctionDef ret */
  ret?: ({[k: string]: string}|null);
}

/** Properties of a GradientDef. */
export declare interface IGradientDef {
  /** GradientDef functionName */
  functionName?: (string|null);

  /** GradientDef gradientFunc */
  gradientFunc?: (string|null);
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// Use the CPU backend for running tests.
import '@tensorflow/tfjs-backend-cpu';
// tslint:disable-next-line: no-imports-from-dist
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
// tslint:disable-next-line:no-imports-from-dist
import {setTestEnvs} from '@tensorflow/tfjs-core/dist/jasmine_util';

// tslint:disable-next-line:no-require-imports
const jasmineCtor = require('jasmine');
// tslint:disable-next-line:no-require-imports

Error.stackTraceLimit = Infinity;

process.on('unhandledRejection', e => {
  throw e;
});

setTestEnvs([{name: 'test-converter', backendName: 'cpu', flags: {}}]);

const unitTests = 'tfjs-converter/src/**/*_test.js';

const runner = new jasmineCtor();
runner.loadConfig({spec_files: [unitTests], random: false});
runner.execute();

/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

export const STRUCTURED_OUTPUTS_MODEL = {
  'modelTopology': {
    'node': [
      {
        'name': 'StatefulPartitionedCall/model/concatenate/concat/axis',
        'op': 'Const',
        'attr': {
          'value': {'tensor': {'dtype': 'DT_INT32', 'tensorShape': {}}},
          'dtype': {'type': 'DT_INT32'}
        }
      },
      {
        'name': 'StatefulPartitionedCall/model/a/MatMul/ReadVariableOp',
        'op': 'Const',
        'attr': {
          'dtype': {'type': 'DT_FLOAT'},
          'value': {
            'tensor': {
              'dtype': 'DT_FLOAT',
              'tensorShape': {'dim': [{'size': '2'}, {'size': '1'}]}
            }
          }
        }
      },
      {
        'name': 'StatefulPartitionedCall/model/b/MatMul/ReadVariableOp',
        'op': 'Const',
        'attr': {
          'value': {
            'tensor': {
              'dtype': 'DT_FLOAT',
              'tensorShape': {'dim': [{'size': '1'}, {'size': '1'}]}
            }
          },
          'dtype': {'type': 'DT_FLOAT'}
        }
      },
      {
        'name': 'input1',
        'op': 'Placeholder',
        'attr': {
          'dtype': {'type': 'DT_FLOAT'},
          'shape': {'shape': {'dim': [{'size': '-1'}, {'size': '1'}]}}
        }
      },
      {
        'name': 'input2',
        'op': 'Placeholder',
        'attr': {
          'dtype': {'type': 'DT_FLOAT'},
          'shape': {'shape': {'dim': [{'size': '-1'}, {'size': '1'}]}}
        }
      },
      {
        'name': 'input3',
        'op': 'Placeholder',
        'attr': {
          'shape': {'shape': {'dim': [{'size': '-1'}, {'size': '1'}]}},
          'dtype': {'type': 'DT_FLOAT'}
        }
      },
      {
        'name': 'StatefulPartitionedCall/model/b/MatMul',
        'op': 'MatMul',
        'input':
            ['input2', 'StatefulPartitionedCall/model/b/MatMul/ReadVariableOp'],
        'device': '/device:CPU:0',
        'attr': {
          'transpose_b': {'b': false},
          'transpose_a': {'b': false},
          'T': {'type': 'DT_FLOAT'}
        }
      },
      {
        'name': 'StatefulPartitionedCall/model/concatenate/concat',
        'op': 'ConcatV2',
        'input': [
          'input1', 'input3',
          'StatefulPartitionedCall/model/concatenate/concat/axis'
        ],
        'attr': {
          'Tidx': {'type': 'DT_INT32'},
          'T': {'type': 'DT_FLOAT'},
          'N': {'i': '2'}
        }
      },
      {
        'name': 'Identity_1',
        'op': 'Identity',
        'input': ['StatefulPartitionedCall/model/b/MatMul'],
        'attr': {'T': {'type': 'DT_FLOAT'}}
      },
      {
        'name': 'StatefulPartitionedCall/model/a/MatMul',
        'op': 'MatMul',
        'input': [
          'StatefulPartitionedCall/model/concatenate/concat',
          'StatefulPartitionedCall/model/a/MatMul/ReadVariableOp'
        ],
        'device': '/device:CPU:0',
        'attr': {
          'T': {'type': 'DT_FLOAT'},
          'transpose_b': {'b': false},
          'transpose_a': {'b': false}
        }
      },
      {
        'name': 'Identity',
        'op': 'Identity',
        'input': ['StatefulPartitionedCall/model/a/MatMul'],
        'attr': {'T': {'type': 'DT_FLOAT'}}
      },
      {
        'name': 'StatefulPartitionedCall/model/c/mul',
        'op': 'Mul',
        'input': [
          'StatefulPartitionedCall/model/a/MatMul',
          'StatefulPartitionedCall/model/b/MatMul'
        ],
        'attr': {'T': {'type': 'DT_FLOAT'}}
      },
      {
        'name': 'Identity_2',
        'op': 'Identity',
        'input': ['StatefulPartitionedCall/model/c/mul'],
        'attr': {'T': {'type': 'DT_FLOAT'}}
      }
    ],
    'library': {},
    'versions': {'producer': 898}
  },
  'format': 'graph-model',
  'generatedBy': '2.7.3',
  'convertedBy': 'TensorFlow.js Converter v1.7.0',
  'weightSpecs': [
    {
      'name': 'StatefulPartitionedCall/model/concatenate/concat/axis',
      'shape': [],
      'dtype': 'int32'
    },
    {
      'name': 'StatefulPartitionedCall/model/a/MatMul/ReadVariableOp',
      'shape': [2, 1],
      'dtype': 'float32'
    },
    {
      'name': 'StatefulPartitionedCall/model/b/MatMul/ReadVariableOp',
      'shape': [1, 1],
      'dtype': 'float32'
    }
  ],
  'weightData': new Uint8Array([
                  0x01, 0x00, 0x00, 0x00, 0x70, 0x3d, 0x72, 0x3e, 0x3d, 0xd2,
                  0x12, 0xbf, 0x0c, 0xfb, 0x94, 0x3e
                ]).buffer,
  'signature': {
    'inputs': {
      'input1:0': {
        'name': 'input1:0',
        'dtype': 'DT_FLOAT',
        'tensorShape': {'dim': [{'size': '-1'}, {'size': '1'}]}
      },
      'input3:0': {
        'name': 'input3:0',
        'dtype': 'DT_FLOAT',
        'tensorShape': {'dim': [{'size': '-1'}, {'size': '1'}]}
      },
      'input2:0': {
        'name': 'input2:0',
        'dtype': 'DT_FLOAT',
        'tensorShape': {'dim': [{'size': '-1'}, {'size': '1'}]}
      }
    },
    'outputs': {
      'Identity_1:0': {
        'name': 'Identity_1:0',
        'dtype': 'DT_FLOAT',
        'tensorShape': {'dim': [{'size': '-1'}, {'size': '1'}]}
      },
      'Identity:0': {
        'name': 'Identity:0',
        'dtype': 'DT_FLOAT',
        'tensorShape': {'dim': [{'size': '-1'}, {'size': '1'}]}
      },
      'Identity_2:0': {
        'name': 'Identity_2:0',
        'dtype': 'DT_FLOAT',
        'tensorShape': {'dim': [{'size': '-1'}, {'size': '1'}]}
      }
    }
  },
  'userDefinedMetadata': {'structuredOutputKeys': ['a', 'b', 'c']}
};

/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
export const HASH_TABLE_MODEL_V2 = {
  modelTopology: {
    node: [
      {
        name: 'unknown_0',
        op: 'Const',
        attr: {
          value: {tensor: {dtype: 'DT_INT32', tensorShape: {}}},
          dtype: {type: 'DT_INT32'}
        }
      },
      {
        name: 'input',
        op: 'Placeholder',
        attr:
            {shape: {shape: {dim: [{size: '-1'}]}}, dtype: {type: 'DT_STRING'}}
      },
      {
        name: 'unknown',
        op: 'Placeholder',
        attr: {shape: {shape: {}}, dtype: {type: 'DT_RESOURCE'}}
      },
      {
        name: 'StatefulPartitionedCall/None_Lookup/LookupTableFindV2',
        op: 'LookupTableFindV2',
        input: ['unknown', 'input', 'unknown_0'],
        attr: {
          Tout: {type: 'DT_INT32'},
          Tin: {type: 'DT_STRING'},
          _has_manual_control_dependencies: {b: true}
        }
      },
      {
        name: 'Identity',
        op: 'Identity',
        input: ['StatefulPartitionedCall/None_Lookup/LookupTableFindV2'],
        attr: {T: {type: 'DT_INT32'}}
      }
    ],
    library: {},
    versions: {producer: 1240}
  },
  format: 'graph-model',
  generatedBy: '2.11.0-dev20220822',
  convertedBy: 'TensorFlow.js Converter v1.7.0',
  weightSpecs: [
    {name: 'unknown_0', shape: [], dtype: 'int32'},
    {name: '114', shape: [2], dtype: 'string'},
    {name: '116', shape: [2], dtype: 'int32'}
  ],
  'weightData':
      new Uint8Array([
        0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0x61, 0x01, 0x00,
        0x00, 0x00, 0x62, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00
      ]).buffer,

  signature: {
    inputs: {
      input: {
        name: 'input:0',
        dtype: 'DT_STRING',
        tensorShape: {dim: [{size: '-1'}]}
      },
      'unknown:0': {
        name: 'unknown:0',
        dtype: 'DT_RESOURCE',
        tensorShape: {},
        resourceId: 66
      }
    },
    outputs: {
      output_0: {
        name: 'Identity:0',
        dtype: 'DT_INT32',
        tensorShape: {dim: [{size: '-1'}]}
      }
    }
  },
  modelInitializer: {
    node: [
      {
        name: 'Func/StatefulPartitionedCall/input_control_node/_0',
        op: 'NoOp',
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name: '114',
        op: 'Const',
        attr: {
          value:
              {tensor: {dtype: 'DT_STRING', tensorShape: {dim: [{size: '2'}]}}},
          _has_manual_control_dependencies: {b: true},
          dtype: {type: 'DT_STRING'}
        }
      },
      {
        name: '116',
        op: 'Const',
        attr: {
          _has_manual_control_dependencies: {b: true},
          dtype: {type: 'DT_INT32'},
          value:
              {tensor: {dtype: 'DT_INT32', tensorShape: {dim: [{size: '2'}]}}}
        }
      },
      {
        name:
            'Func/StatefulPartitionedCall/StatefulPartitionedCall/input_control_node/_9',
        op: 'NoOp',
        input: ['^Func/StatefulPartitionedCall/input_control_node/_0'],
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name: 'StatefulPartitionedCall/StatefulPartitionedCall/hash_table',
        op: 'HashTableV2',
        input: [
          '^Func/StatefulPartitionedCall/StatefulPartitionedCall/input_control_node/_9'
        ],
        attr: {
          container: {s: ''},
          use_node_name_sharing: {b: true},
          _has_manual_control_dependencies: {b: true},
          shared_name: {s: 'OTVfbG9hZF8xXzUy'},
          value_dtype: {type: 'DT_INT32'},
          key_dtype: {type: 'DT_STRING'}
        }
      },
      {
        name:
            'Func/StatefulPartitionedCall/StatefulPartitionedCall/output_control_node/_11',
        op: 'NoOp',
        input: ['^StatefulPartitionedCall/StatefulPartitionedCall/hash_table'],
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name: 'Func/StatefulPartitionedCall/output_control_node/_2',
        op: 'NoOp',
        input: [
          '^Func/StatefulPartitionedCall/StatefulPartitionedCall/output_control_node/_11'
        ],
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name: 'StatefulPartitionedCall/StatefulPartitionedCall/NoOp',
        op: 'NoOp',
        input: ['^StatefulPartitionedCall/StatefulPartitionedCall/hash_table'],
        attr: {
          _acd_function_control_output: {b: true},
          _has_manual_control_dependencies: {b: true}
        }
      },
      {
        name: 'StatefulPartitionedCall/StatefulPartitionedCall/Identity',
        op: 'Identity',
        input: [
          'StatefulPartitionedCall/StatefulPartitionedCall/hash_table',
          '^StatefulPartitionedCall/StatefulPartitionedCall/NoOp'
        ],
        attr: {T: {type: 'DT_RESOURCE'}}
      },
      {
        name: 'Func/StatefulPartitionedCall/StatefulPartitionedCall/output/_10',
        op: 'Identity',
        input: ['StatefulPartitionedCall/StatefulPartitionedCall/Identity'],
        attr: {T: {type: 'DT_RESOURCE'}}
      },
      {
        name: 'StatefulPartitionedCall/NoOp',
        op: 'NoOp',
        input: [
          '^Func/StatefulPartitionedCall/StatefulPartitionedCall/output_control_node/_11'
        ],
        attr: {
          _has_manual_control_dependencies: {b: true},
          _acd_function_control_output: {b: true}
        }
      },
      {
        name: 'StatefulPartitionedCall/Identity',
        op: 'Identity',
        input: [
          'Func/StatefulPartitionedCall/StatefulPartitionedCall/output/_10',
          '^StatefulPartitionedCall/NoOp'
        ],
        attr: {T: {type: 'DT_RESOURCE'}}
      },
      {
        name: 'Func/StatefulPartitionedCall/output/_1',
        op: 'Identity',
        input: ['StatefulPartitionedCall/Identity'],
        attr: {
          T: {type: 'DT_RESOURCE'},
          _has_manual_control_dependencies: {b: true}
        }
      },
      {
        name: 'Func/StatefulPartitionedCall_1/input_control_node/_3',
        op: 'NoOp',
        input: ['^114', '^116', '^Func/StatefulPartitionedCall/output/_1'],
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name: 'Func/StatefulPartitionedCall_1/input/_4',
        op: 'Identity',
        input: [
          'Func/StatefulPartitionedCall/output/_1',
          '^Func/StatefulPartitionedCall_1/input_control_node/_3'
        ],
        attr: {T: {type: 'DT_RESOURCE'}}
      },
      {
        name:
            'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input_control_node/_12',
        op: 'NoOp',
        input: ['^Func/StatefulPartitionedCall_1/input_control_node/_3'],
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name:
            'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_13',
        op: 'Identity',
        input: [
          'Func/StatefulPartitionedCall_1/input/_4',
          '^Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input_control_node/_12'
        ],
        attr: {T: {type: 'DT_RESOURCE'}}
      },
      {
        name: 'Func/StatefulPartitionedCall_1/input/_5',
        op: 'Identity',
        input: ['114', '^Func/StatefulPartitionedCall_1/input_control_node/_3'],
        attr: {T: {type: 'DT_STRING'}}
      },
      {
        name:
            'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_14',
        op: 'Identity',
        input: [
          'Func/StatefulPartitionedCall_1/input/_5',
          '^Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input_control_node/_12'
        ],
        attr: {T: {type: 'DT_STRING'}}
      },
      {
        name: 'Func/StatefulPartitionedCall_1/input/_6',
        op: 'Identity',
        input: ['116', '^Func/StatefulPartitionedCall_1/input_control_node/_3'],
        attr: {T: {type: 'DT_INT32'}}
      },
      {
        name:
            'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_15',
        op: 'Identity',
        input: [
          'Func/StatefulPartitionedCall_1/input/_6',
          '^Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input_control_node/_12'
        ],
        attr: {T: {type: 'DT_INT32'}}
      },
      {
        name:
            'StatefulPartitionedCall_1/StatefulPartitionedCall/key_value_init94/LookupTableImportV2',
        op: 'LookupTableImportV2',
        input: [
          'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_13',
          'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_14',
          'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/input/_15'
        ],
        attr: {
          Tout: {type: 'DT_INT32'},
          Tin: {type: 'DT_STRING'},
          _has_manual_control_dependencies: {b: true}
        }
      },
      {
        name:
            'Func/StatefulPartitionedCall_1/StatefulPartitionedCall/output_control_node/_17',
        op: 'NoOp',
        input: [
          '^StatefulPartitionedCall_1/StatefulPartitionedCall/key_value_init94/LookupTableImportV2'
        ],
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name: 'Func/StatefulPartitionedCall_1/output_control_node/_8',
        op: 'NoOp',
        input: [
          '^Func/StatefulPartitionedCall_1/StatefulPartitionedCall/output_control_node/_17'
        ],
        attr: {_has_manual_control_dependencies: {b: true}}
      },
      {
        name: 'NoOp',
        op: 'NoOp',
        input: [
          '^Func/StatefulPartitionedCall/output_control_node/_2',
          '^Func/StatefulPartitionedCall_1/output_control_node/_8'
        ],
        attr: {
          _has_manual_control_dependencies: {b: true},
          _acd_function_control_output: {b: true}
        }
      },
      {
        name: 'Identity',
        op: 'Identity',
        input: [
          'Func/StatefulPartitionedCall/output/_1',
          '^Func/StatefulPartitionedCall_1/output_control_node/_8', '^NoOp'
        ],
        attr: {T: {type: 'DT_RESOURCE'}}
      }
    ],
    versions: {producer: 1240}
  },
  initializerSignature: {
    outputs: {
      'Identity:0': {
        name: 'Identity:0',
        dtype: 'DT_RESOURCE',
        tensorShape: {},
        resourceId: 66
      }
    }
  }
};

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {dispose, InferenceModel, io, ModelPredictConfig, NamedTensorMap, Tensor, util} from '@tensorflow/tfjs-core';

import * as tensorflow from '../data/compiled_api';
import {NamedTensorsMap, TensorInfo} from '../data/types';
import {OperationMapper} from '../operations/operation_mapper';

import {GraphExecutor} from './graph_executor';
import {ResourceManager} from './resource_manager';
// tslint:disable-next-line: no-imports-from-dist
import {decodeWeightsStream} from '@tensorflow/tfjs-core/dist/io/io_utils';

export const TFHUB_SEARCH_PARAM = '?tfjs-format=file';
export const DEFAULT_MODEL_NAME = 'model.json';
type Url = string|io.IOHandler|io.IOHandlerSync;
type UrlIOHandler<T extends Url> = T extends string ? io.IOHandler : T;

/**
 * A `tf.GraphModel` is a directed, acyclic graph built from a
 * SavedModel GraphDef and allows inference execution.
 *
 * A `tf.GraphModel` can only be created by loading from a model converted from
 * a [TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model) using
 * the command line converter tool and loaded via `tf.loadGraphModel`.
 *
 * @doc {heading: 'Models', subheading: 'Classes'}
 */
export class GraphModel<ModelURL extends Url = string | io.IOHandler> implements
    InferenceModel {
  private executor: GraphExecutor;
  private version = 'n/a';
  private handler: UrlIOHandler<ModelURL>;
  private artifacts: io.ModelArtifacts;
  private initializer: GraphExecutor;
  private resourceIdToCapturedInput: {[key: number]: Tensor};
  private resourceManager: ResourceManager;
  private signature: tensorflow.ISignatureDef;
  private initializerSignature: tensorflow.ISignatureDef;
  private structuredOutputKeys: string[];
  private readonly io: typeof io;

  // Returns the version information for the tensorflow model GraphDef.
  get modelVersion(): string {
    return this.version;
  }

  get inputNodes(): string[] {
    return this.executor.inputNodes;
  }

  get outputNodes(): string[] {
    return this.executor.outputNodes;
  }

  get inputs(): TensorInfo[] {
    return this.executor.inputs;
  }

  get outputs(): TensorInfo[] {
    return this.executor.outputs;
  }

  get weights(): NamedTensorsMap {
    return this.executor.weightMap;
  }

  get metadata(): {} {
    return this.artifacts.userDefinedMetadata;
  }

  get modelSignature(): {} {
    return this.signature;
  }

  get modelStructuredOutputKeys(): {} {
    return this.structuredOutputKeys;
  }

  /**
   * @param modelUrl url for the model, or an `io.IOHandler`.
   * @param weightManifestUrl url for the weight file generated by
   * scripts/convert.py script.
   * @param requestOption options for Request, which allows to send credentials
   * and custom headers.
   * @param onProgress Optional, progress callback function, fired periodically
   * before the load is completed.
   */
  constructor(
      private modelUrl: ModelURL, private loadOptions: io.LoadOptions = {},
      tfio = io) {
    this.io = tfio;
    if (loadOptions == null) {
      this.loadOptions = {};
    }
    this.resourceManager = new ResourceManager();
  }

  private findIOHandler() {
    type IOHandler = UrlIOHandler<ModelURL>;
    const path = this.modelUrl;
    if ((path as io.IOHandler).load != null) {
      // Path is an IO Handler.
      this.handler = path as IOHandler;
    } else if (this.loadOptions.requestInit != null) {
      this.handler = this.io.browserHTTPRequest(
                         path as string, this.loadOptions) as IOHandler;
    } else {
      const handlers =
          this.io.getLoadHandlers(path as string, this.loadOptions);
      if (handlers.length === 0) {
        // For backward compatibility: if no load handler can be found,
        // assume it is a relative http path.
        handlers.push(
            this.io.browserHTTPRequest(path as string, this.loadOptions));
      } else if (handlers.length > 1) {
        throw new Error(
            `Found more than one (${handlers.length}) load handlers for ` +
            `URL '${[path]}'`);
      }
      this.handler = handlers[0] as IOHandler;
    }
  }

  /**
   * Loads the model and weight files, construct the in memory weight map and
   * compile the inference graph.
   */
  load(): UrlIOHandler<ModelURL> extends io.IOHandlerSync? boolean:
                                             Promise<boolean> {
    type IOHandler = UrlIOHandler<ModelURL>;
    this.findIOHandler();
    if (this.handler.load == null) {
      throw new Error(
          'Cannot proceed with model loading because the IOHandler provided ' +
          'does not have the `load` method implemented.');
    }

    type Result =
        IOHandler extends io.IOHandlerSync ? boolean : Promise<boolean>;

    const loadResult = this.handler.load() as ReturnType<IOHandler['load']>;
    if (util.isPromise(loadResult)) {
      return loadResult.then(artifacts => {
        if (artifacts.getWeightStream == null) {
          return this.loadSync(artifacts);
        }
        return this.loadStreaming(artifacts);
      }) as Result;
    }

    return this.loadSync(loadResult) as Result;
  }

  /**
   * Synchronously construct the in memory weight map and
   * compile the inference graph.
   *
   * @doc {heading: 'Models', subheading: 'Classes', ignoreCI: true}
   */
  loadSync(artifacts: io.ModelArtifacts) {
    const weightMap = this.io.decodeWeights(
        artifacts.weightData, artifacts.weightSpecs);

    return this.loadWithWeightMap(artifacts, weightMap);
  }

  private async loadStreaming(artifacts: io.ModelArtifacts): Promise<boolean> {
    if (artifacts.getWeightStream == null) {
      throw new Error('Model artifacts missing streamWeights function');
    }

    const weightMap = await decodeWeightsStream(
      artifacts.getWeightStream(), artifacts.weightSpecs);

    return this.loadWithWeightMap(artifacts, weightMap);
  }

  private loadWithWeightMap(artifacts: io.ModelArtifacts,
                            weightMap: NamedTensorMap) {
    this.artifacts = artifacts;
    const graph = this.artifacts.modelTopology as tensorflow.IGraphDef;

    let signature = this.artifacts.signature;
    if (this.artifacts.userDefinedMetadata != null) {
      const metadata = this.artifacts.userDefinedMetadata;
      if (metadata.signature != null) {
        signature = metadata.signature;
      }

      if (metadata.structuredOutputKeys != null) {
        this.structuredOutputKeys = metadata.structuredOutputKeys as string[];
      }
    }
    this.signature = signature;

    this.version = `${graph.versions.producer}.${graph.versions.minConsumer}`;
    this.executor = new GraphExecutor(
        OperationMapper.Instance.transformGraph(graph, this.signature));
    this.executor.weightMap = this.convertTensorMapToTensorsMap(weightMap);
    // Attach a model-level resourceManager to each executor to share resources,
    // such as `HashTable`.
    this.executor.resourceManager = this.resourceManager;

    if (artifacts.modelInitializer != null &&
        (artifacts.modelInitializer as tensorflow.IGraphDef).node != null) {
      const initializer =
          OperationMapper.Instance.transformGraph(artifacts.modelInitializer);
      this.initializer = new GraphExecutor(initializer);
      this.initializer.weightMap = this.executor.weightMap;
      // Attach a model-level resourceManager to the initializer, the
      // hashTables created from when executing the initializer will be stored
      // in the resourceManager.
      this.initializer.resourceManager = this.resourceManager;
      this.initializerSignature = artifacts.initializerSignature;
    }

    return true;
  }

  /**
   * Save the configuration and/or weights of the GraphModel.
   *
   * An `IOHandler` is an object that has a `save` method of the proper
   * signature defined. The `save` method manages the storing or
   * transmission of serialized data ("artifacts") that represent the
   * model's topology and weights onto or via a specific medium, such as
   * file downloads, local storage, IndexedDB in the web browser and HTTP
   * requests to a server. TensorFlow.js provides `IOHandler`
   * implementations for a number of frequently used saving mediums, such as
   * `tf.io.browserDownloads` and `tf.io.browserLocalStorage`. See `tf.io`
   * for more details.
   *
   * This method also allows you to refer to certain types of `IOHandler`s
   * as URL-like string shortcuts, such as 'localstorage://' and
   * 'indexeddb://'.
   *
   * Example 1: Save `model`'s topology and weights to browser [local
   * storage](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage);
   * then load it back.
   *
   * ```js
   * const modelUrl =
   *    'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json';
   * const model = await tf.loadGraphModel(modelUrl);
   * const zeros = tf.zeros([1, 224, 224, 3]);
   * model.predict(zeros).print();
   *
   * const saveResults = await model.save('localstorage://my-model-1');
   *
   * const loadedModel = await tf.loadGraphModel('localstorage://my-model-1');
   * console.log('Prediction from loaded model:');
   * model.predict(zeros).print();
   * ```
   *
   * @param handlerOrURL An instance of `IOHandler` or a URL-like,
   * scheme-based string shortcut for `IOHandler`.
   * @param config Options for saving the model.
   * @returns A `Promise` of `SaveResult`, which summarizes the result of
   * the saving, such as byte sizes of the saved artifacts for the model's
   *   topology and weight values.
   *
   * @doc {heading: 'Models', subheading: 'Classes', ignoreCI: true}
   */
  async save(handlerOrURL: io.IOHandler|string, config?: io.SaveConfig):
      Promise<io.SaveResult> {
    if (typeof handlerOrURL === 'string') {
      const handlers = this.io.getSaveHandlers(handlerOrURL);
      if (handlers.length === 0) {
        throw new Error(
            `Cannot find any save handlers for URL '${handlerOrURL}'`);
      } else if (handlers.length > 1) {
        throw new Error(
            `Found more than one (${handlers.length}) save handlers for ` +
            `URL '${handlerOrURL}'`);
      }
      handlerOrURL = handlers[0];
    }
    if (handlerOrURL.save == null) {
      throw new Error(
          'GraphModel.save() cannot proceed because the IOHandler ' +
          'provided does not have the `save` attribute defined.');
    }

    return handlerOrURL.save(this.artifacts);
  }

  private addStructuredOutputNames(outputTensors: Tensor|Tensor[]) {
    if (this.structuredOutputKeys) {
      const outputTensorsArray =
          outputTensors instanceof Tensor ? [outputTensors] : outputTensors;
      const outputTensorMap: NamedTensorMap = {};

      outputTensorsArray.forEach(
          (outputTensor, i) => outputTensorMap[this.structuredOutputKeys[i]] =
              outputTensor);

      return outputTensorMap;
    }
    return outputTensors;
  }

  /**
   * Execute the inference for the input tensors.
   *
   * @param input The input tensors, when there is single input for the model,
   * inputs param should be a `tf.Tensor`. For models with multiple inputs,
   * inputs params should be in either `tf.Tensor`[] if the input order is
   * fixed, or otherwise NamedTensorMap format.
   *
   * For model with multiple inputs, we recommend you use NamedTensorMap as the
   * input type, if you use `tf.Tensor`[], the order of the array needs to
   * follow the
   * order of inputNodes array. @see {@link GraphModel.inputNodes}
   *
   * You can also feed any intermediate nodes using the NamedTensorMap as the
   * input type. For example, given the graph
   *    InputNode => Intermediate => OutputNode,
   * you can execute the subgraph Intermediate => OutputNode by calling
   *    model.execute('IntermediateNode' : tf.tensor(...));
   *
   * This is useful for models that uses tf.dynamic_rnn, where the intermediate
   * state needs to be fed manually.
   *
   * For batch inference execution, the tensors for each input need to be
   * concatenated together. For example with mobilenet, the required input shape
   * is [1, 244, 244, 3], which represents the [batch, height, width, channel].
   * If we are provide a batched data of 100 images, the input tensor should be
   * in the shape of [100, 244, 244, 3].
   *
   * @param config Prediction configuration for specifying the batch size.
   * Currently the batch size option is ignored for graph model.
   *
   * @returns Inference result tensors. If the model is converted and it
   * originally had structured_outputs in tensorflow, then a NamedTensorMap
   * will be returned matching the structured_outputs. If no structured_outputs
   * are present, the output will be single `tf.Tensor` if the model has single
   * output node, otherwise Tensor[].
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  predict(inputs: Tensor|Tensor[]|NamedTensorMap, config?: ModelPredictConfig):
      Tensor|Tensor[]|NamedTensorMap {
    const outputTensors = this.execute(inputs, this.outputNodes);
    return this.addStructuredOutputNames(outputTensors);
  }

  /**
   * Execute the inference for the input tensors in async fashion, use this
   * method when your model contains control flow ops.
   *
   * @param input The input tensors, when there is single input for the model,
   * inputs param should be a `tf.Tensor`. For models with mutliple inputs,
   * inputs params should be in either `tf.Tensor`[] if the input order is
   * fixed, or otherwise NamedTensorMap format.
   *
   * For model with multiple inputs, we recommend you use NamedTensorMap as the
   * input type, if you use `tf.Tensor`[], the order of the array needs to
   * follow the
   * order of inputNodes array. @see {@link GraphModel.inputNodes}
   *
   * You can also feed any intermediate nodes using the NamedTensorMap as the
   * input type. For example, given the graph
   *    InputNode => Intermediate => OutputNode,
   * you can execute the subgraph Intermediate => OutputNode by calling
   *    model.execute('IntermediateNode' : tf.tensor(...));
   *
   * This is useful for models that uses tf.dynamic_rnn, where the intermediate
   * state needs to be fed manually.
   *
   * For batch inference execution, the tensors for each input need to be
   * concatenated together. For example with mobilenet, the required input shape
   * is [1, 244, 244, 3], which represents the [batch, height, width, channel].
   * If we are provide a batched data of 100 images, the input tensor should be
   * in the shape of [100, 244, 244, 3].
   *
   * @param config Prediction configuration for specifying the batch size.
   * Currently the batch size option is ignored for graph model.
   *
   * @returns A Promise of inference result tensors. If the model is converted
   * and it originally had structured_outputs in tensorflow, then a
   * NamedTensorMap will be returned matching the structured_outputs. If no
   * structured_outputs are present, the output will be single `tf.Tensor` if
   * the model has single output node, otherwise Tensor[].
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  async predictAsync(
      inputs: Tensor|Tensor[]|NamedTensorMap,
      config?: ModelPredictConfig): Promise<Tensor|Tensor[]|NamedTensorMap> {
    const outputTensors = await this.executeAsync(inputs, this.outputNodes);
    return this.addStructuredOutputNames(outputTensors);
  }

  private normalizeInputs(inputs: Tensor|Tensor[]|
                          NamedTensorMap): NamedTensorMap {
    if (!(inputs instanceof Tensor) && !Array.isArray(inputs)) {
      // The input is already a NamedTensorMap.
      const signatureInputs = this.signature?.inputs;
      if (signatureInputs != null) {
        for (const input in signatureInputs) {
          const tensor = signatureInputs[input];
          if (tensor.resourceId != null) {
            inputs[input] = this.resourceIdToCapturedInput[tensor.resourceId];
          }
        }
      }
      return inputs;
    }
    inputs = Array.isArray(inputs) ? inputs : [inputs];

    const numCapturedInputs =
        Object.keys(this.resourceIdToCapturedInput).length;
    if (inputs.length + numCapturedInputs !== this.inputNodes.length) {
      throw new Error(`Input tensor count mismatch, the graph model has ${
          this.inputNodes.length -
          numCapturedInputs} non-resource placeholders, while there are ${
          inputs.length} input tensors provided.`);
    }

    let inputIndex = 0;
    return this.inputNodes.reduce((map, inputName) => {
      const resourceId = this.signature?.inputs?.[inputName]?.resourceId;
      if (resourceId != null) {
        map[inputName] = this.resourceIdToCapturedInput[resourceId];
      } else {
        map[inputName] = (inputs as Tensor[])[inputIndex++];
      }
      return map;
    }, {} as NamedTensorMap);
  }

  private normalizeOutputs(outputs: string|string[]): string[] {
    outputs = outputs || this.outputNodes;
    return !Array.isArray(outputs) ? [outputs] : outputs;
  }

  private executeInitializerGraph() {
    if (this.initializer == null) {
      return [];
    }
    if (this.initializerSignature == null) {
      return this.initializer.execute({}, []);
    } else {
      return this.initializer.execute(
          {}, Object.keys(this.initializerSignature.outputs));
    }
  }

  private async executeInitializerGraphAsync() {
    if (this.initializer == null) {
      return [];
    }
    if (this.initializerSignature == null) {
      return this.initializer.executeAsync({}, []);
    } else {
      return this.initializer.executeAsync(
          {}, Object.keys(this.initializerSignature.outputs));
    }
  }

  private setResourceIdToCapturedInput(outputs: Tensor[]) {
    this.resourceIdToCapturedInput = {};

    if (this.initializerSignature) {
      const signatureOutputs = this.initializerSignature.outputs;
      const outputNames = Object.keys(signatureOutputs);
      for (let i = 0; i < outputNames.length; i++) {
        const outputName = outputNames[i];
        const tensorInfo = signatureOutputs[outputName];
        this.resourceIdToCapturedInput[tensorInfo.resourceId] = outputs[i];
      }
    }
  }

  /**
   * Executes inference for the model for given input tensors.
   * @param inputs tensor, tensor array or tensor map of the inputs for the
   * model, keyed by the input node names.
   * @param outputs output node name from the TensorFlow model, if no
   * outputs are specified, the default outputs of the model would be used.
   * You can inspect intermediate nodes of the model by adding them to the
   * outputs array.
   *
   * @returns A single tensor if provided with a single output or no outputs
   * are provided and there is only one default output, otherwise return a
   * tensor array. The order of the tensor array is the same as the outputs
   * if provided, otherwise the order of outputNodes attribute of the model.
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  execute(inputs: Tensor|Tensor[]|NamedTensorMap, outputs?: string|string[]):
      Tensor|Tensor[] {
    if (this.resourceIdToCapturedInput == null) {
      this.setResourceIdToCapturedInput(this.executeInitializerGraph());
    }
    inputs = this.normalizeInputs(inputs);
    outputs = this.normalizeOutputs(outputs);
    const result = this.executor.execute(inputs, outputs);
    return result.length > 1 ? result : result[0];
  }

  /**
   * Executes inference for the model for given input tensors in async
   * fashion, use this method when your model contains control flow ops.
   * @param inputs tensor, tensor array or tensor map of the inputs for the
   * model, keyed by the input node names.
   * @param outputs output node name from the TensorFlow model, if no outputs
   * are specified, the default outputs of the model would be used. You can
   * inspect intermediate nodes of the model by adding them to the outputs
   * array.
   *
   * @returns A Promise of single tensor if provided with a single output or
   * no outputs are provided and there is only one default output, otherwise
   * return a tensor map.
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  async executeAsync(
      inputs: Tensor|Tensor[]|NamedTensorMap,
      outputs?: string|string[]): Promise<Tensor|Tensor[]> {
    if (this.resourceIdToCapturedInput == null) {
      this.setResourceIdToCapturedInput(
          await this.executeInitializerGraphAsync());
    }
    inputs = this.normalizeInputs(inputs);
    outputs = this.normalizeOutputs(outputs);
    const result = await this.executor.executeAsync(inputs, outputs);
    return result.length > 1 ? result : result[0];
  }

  /**
   * Get intermediate tensors for model debugging mode (flag
   * KEEP_INTERMEDIATE_TENSORS is true).
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  getIntermediateTensors(): NamedTensorsMap {
    return this.executor.getIntermediateTensors();
  }

  /**
   * Dispose intermediate tensors for model debugging mode (flag
   * KEEP_INTERMEDIATE_TENSORS is true).
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  disposeIntermediateTensors() {
    this.executor.disposeIntermediateTensors();
  }

  private convertTensorMapToTensorsMap(map: NamedTensorMap): NamedTensorsMap {
    return Object.keys(map).reduce((newMap: NamedTensorsMap, key) => {
      newMap[key] = [map[key]];
      return newMap;
    }, {});
  }

  /**
   * Releases the memory used by the weight tensors and resourceManager.
   *
   * @doc {heading: 'Models', subheading: 'Classes'}
   */
  dispose() {
    this.executor.dispose();

    if (this.initializer) {
      this.initializer.dispose();
      if (this.resourceIdToCapturedInput) {
        dispose(this.resourceIdToCapturedInput);
      }
    }

    this.resourceManager.dispose();
  }
}

/**
 * Load a graph model given a URL to the model definition.
 *
 * Example of loading MobileNetV2 from a URL and making a prediction with a
 * zeros input:
 *
 * ```js
 * const modelUrl =
 *    'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json';
 * const model = await tf.loadGraphModel(modelUrl);
 * const zeros = tf.zeros([1, 224, 224, 3]);
 * model.predict(zeros).print();
 * ```
 *
 * Example of loading MobileNetV2 from a TF Hub URL and making a prediction
 * with a zeros input:
 *
 * ```js
 * const modelUrl =
 *    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
 * const model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
 * const zeros = tf.zeros([1, 224, 224, 3]);
 * model.predict(zeros).print();
 * ```
 * @param modelUrl The url or an `io.IOHandler` that loads the model.
 * @param options Options for the HTTP request, which allows to send
 *     credentials
 *    and custom headers.
 *
 * @doc {heading: 'Models', subheading: 'Loading'}
 */
export async function loadGraphModel(
    modelUrl: string|io.IOHandler, options: io.LoadOptions = {},
    tfio = io): Promise<GraphModel> {
  if (modelUrl == null) {
    throw new Error(
        'modelUrl in loadGraphModel() cannot be null. Please provide a url ' +
        'or an IOHandler that loads the model');
  }
  if (options == null) {
    options = {};
  }

  if (options.fromTFHub && typeof modelUrl === 'string') {
    modelUrl = getTFHubUrl(modelUrl);
  }
  const model = new GraphModel(modelUrl, options, tfio);
  await model.load();
  return model;
}

/**
 * Load a graph model given a synchronous IO handler with a 'load' method.
 *
 * @param modelSource The `io.IOHandlerSync` that loads the model, or the
 *     `io.ModelArtifacts` that encode the model, or a tuple of
 *     `[io.ModelJSON, ArrayBuffer]` of which the first element encodes the
 *      model and the second contains the weights.
 *
 * @doc {heading: 'Models', subheading: 'Loading'}
 */
export function loadGraphModelSync(
    modelSource: io.IOHandlerSync|
    io.ModelArtifacts|[io.ModelJSON, /* Weights */ ArrayBuffer]):
    GraphModel<io.IOHandlerSync> {
  if (modelSource == null) {
    throw new Error(
        'modelUrl in loadGraphModelSync() cannot be null. Please provide ' +
        'model artifacts or an IOHandler that loads the model');
  }

  let ioHandler: io.IOHandlerSync;
  if (modelSource instanceof Array) {
    const [modelJSON, weights] = modelSource;
    if (!modelJSON) {
      throw new Error('modelJSON must be the first element of the array');
    }
    if (!weights || !(weights instanceof ArrayBuffer)) {
      throw new Error(
          'An ArrayBuffer of weights must be the second element of' +
          ' the array');
    }
    if (!('modelTopology' in modelJSON)) {
      throw new Error('Model JSON is missing \'modelTopology\'');
    }
    if (!('weightsManifest' in modelJSON)) {
      throw new Error('Model JSON is missing \'weightsManifest\'');
    }

    const weightSpecs = io.getWeightSpecs(modelJSON.weightsManifest);
    const modelArtifacts =
        io.getModelArtifactsForJSONSync(modelJSON, weightSpecs, weights);
    ioHandler = io.fromMemorySync(modelArtifacts);
  } else if ('load' in modelSource) {
    // Then modelSource is already an IOHandlerSync.
    ioHandler = modelSource;
  } else if (
      'modelTopology' in modelSource && 'weightSpecs' in modelSource &&
      'weightData' in modelSource) {
    // modelSource is of type ModelArtifacts.
    ioHandler = io.fromMemorySync(modelSource);
  } else {
    throw new Error('Unknown model format');
  }

  const model = new GraphModel(ioHandler);
  model.load();
  return model;
}

function getTFHubUrl(modelUrl: string): string {
  if (!modelUrl.endsWith('/')) {
    modelUrl = (modelUrl) + '/';
  }
  return `${modelUrl}${DEFAULT_MODEL_NAME}${TFHUB_SEARCH_PARAM}`;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Tensor} from '@tensorflow/tfjs-core';

import {NamedTensorsMap, TensorArrayMap, TensorListMap} from '../data/types';

/**
 *
 */
export interface FunctionExecutor {
  executeFunctionAsync(
      inputs: Tensor[], tensorArrayMap: TensorArrayMap,
      tensorListMap: TensorListMap): Promise<Tensor[]>;
  weightMap: NamedTensorsMap;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import {HashTableMap, NamedTensorMap} from '../data/types';
import {HashTable} from './hash_table';

/**
 * Contains global resources of a model.
 */
export class ResourceManager {
  constructor(
      readonly hashTableNameToHandle: NamedTensorMap = {},
      readonly hashTableMap: HashTableMap = {}) {}

  /**
   * Register a `HashTable` in the resource manager.
   *
   * The `HashTable` can be retrieved by `resourceManager.getHashTableById`,
   * where id is the table handle tensor's id.
   *
   * @param name Op node name that creates the `HashTable`.
   * @param hashTable The `HashTable` to be added to resource manager.
   */
  addHashTable(name: string, hashTable: HashTable) {
    this.hashTableNameToHandle[name] = hashTable.handle;
    this.hashTableMap[hashTable.id] = hashTable;
  }

  /**
   * Get the table handle by node name.
   * @param name Op node name that creates the `HashTable`. This name is also
   *     used in the inputs list of lookup and import `HashTable` ops.
   */
  getHashTableHandleByName(name: string) {
    return this.hashTableNameToHandle[name];
  }

  /**
   * Get the actual `HashTable` by its handle tensor's id.
   * @param id The id of the handle tensor.
   */
  getHashTableById(id: number): HashTable {
    return this.hashTableMap[id];
  }

  /**
   * Dispose `ResourceManager`, including its hashTables and tensors in them.
   */
  dispose() {
    for (const key in this.hashTableMap) {
      this.hashTableMap[key].clearAndClose();
      delete this.hashTableMap[key];
    }

    for (const name in this.hashTableNameToHandle) {
      this.hashTableNameToHandle[name].dispose();
      delete this.hashTableNameToHandle[name];
    }
  }
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import {DataType, keep, scalar, stack, Tensor, tidy, unstack, util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

/**
 * Hashtable contains a set of tensors, which can be accessed by key.
 */
export class HashTable {
  readonly handle: Tensor;

  // tslint:disable-next-line: no-any
  private tensorMap: Map<any, Tensor>;

  get id() {
    return this.handle.id;
  }

  /**
   * Constructor of HashTable. Creates a hash table.
   *
   * @param keyDType `dtype` of the table keys.
   * @param valueDType `dtype` of the table values.
   */
  constructor(readonly keyDType: DataType, readonly valueDType: DataType) {
    this.handle = scalar(0);
    // tslint:disable-next-line: no-any
    this.tensorMap = new Map<any, Tensor>();

    keep(this.handle);
  }

  /**
   * Dispose the tensors and handle and clear the hashtable.
   */
  clearAndClose() {
    this.tensorMap.forEach(value => value.dispose());
    this.tensorMap.clear();
    this.handle.dispose();
  }

  /**
   * The number of items in the hash table.
   */
  size(): number {
    return this.tensorMap.size;
  }

  /**
   * The number of items in the hash table as a rank-0 tensor.
   */
  tensorSize(): Tensor {
    return tfOps.scalar(this.size(), 'int32');
  }

  /**
   * Replaces the contents of the table with the specified keys and values.
   * @param keys Keys to store in the hashtable.
   * @param values Values to store in the hashtable.
   */
  async import(keys: Tensor, values: Tensor): Promise<Tensor> {
    this.checkKeyAndValueTensor(keys, values);

    // We only store the primitive values of the keys, this allows lookup
    // to be O(1).
    const $keys = await keys.data();

    // Clear the hashTable before inserting new values.
    this.tensorMap.forEach(value => value.dispose());
    this.tensorMap.clear();

    return tidy(() => {
      const $values = unstack(values);

      const keysLength = $keys.length;
      const valuesLength = $values.length;

      util.assert(
          keysLength === valuesLength,
          () => `The number of elements doesn't match, keys has ` +
              `${keysLength} elements, the values has ${valuesLength} ` +
              `elements.`);

      for (let i = 0; i < keysLength; i++) {
        const key = $keys[i];
        const value = $values[i];

        keep(value);
        this.tensorMap.set(key, value);
      }

      return this.handle;
    });
  }

  /**
   * Looks up keys in a hash table, outputs the corresponding values.
   *
   * Performs batch lookups, for every element in the key tensor, `find`
   * stacks the corresponding value into the return tensor.
   *
   * If an element is not present in the table, the given `defaultValue` is
   * used.
   *
   * @param keys Keys to look up. Must have the same type as the keys of the
   *     table.
   * @param defaultValue The scalar `defaultValue` is the value output for keys
   *     not present in the table. It must also be of the same type as the
   *     table values.
   */
  async find(keys: Tensor, defaultValue: Tensor): Promise<Tensor> {
    this.checkKeyAndValueTensor(keys, defaultValue);

    const $keys = await keys.data();

    return tidy(() => {
      const result: Tensor[] = [];

      for (let i = 0; i < $keys.length; i++) {
        const key = $keys[i];

        const value = this.findWithDefault(key, defaultValue);
        result.push(value);
      }

      return stack(result);
    });
  }

  // tslint:disable-next-line: no-any
  private findWithDefault(key: any, defaultValue: Tensor): Tensor {
    const result = this.tensorMap.get(key);

    return result != null ? result : defaultValue;
  }

  private checkKeyAndValueTensor(key: Tensor, value: Tensor) {
    if (key.dtype !== this.keyDType) {
      throw new Error(
          `Expect key dtype ${this.keyDType}, but got ` +
          `${key.dtype}`);
    }

    if (value.dtype !== this.valueDType) {
      throw new Error(
          `Expect value dtype ${this.valueDType}, but got ` +
          `${value.dtype}`);
    }
  }
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {DataType, env, keep, NamedTensorMap, Tensor, tidy, util} from '@tensorflow/tfjs-core';

import {ISignatureDef} from '../data/compiled_api';
import {NamedTensorsMap, TensorArrayMap, TensorInfo, TensorListMap} from '../data/types';
import {getNodeNameAndIndex, getParamValue, getTensor, getTensorsForCurrentContext, parseNodeName} from '../operations/executors/utils';
import {executeOp} from '../operations/operation_executor';
import {Graph, Node} from '../operations/types';

import {ExecutionContext, ExecutionContextInfo} from './execution_context';
import {getExecutionSubgraph, getNodeLiveUntilMap, getNodesInTopologicalOrder, isControlFlow} from './model_analysis';
import {ResourceManager} from './resource_manager';
import {FunctionExecutor} from './types';

interface NodeWithContexts {
  contexts: ExecutionContextInfo[];
  node: Node;
}

export class GraphExecutor implements FunctionExecutor {
  private compiledMap = new Map<string, ReturnType<typeof this.compile>>();
  private parseNodeNameCache = new Map<string, [string, number, string?]>();
  private _weightMap: NamedTensorsMap = {};
  private _weightIds: number[];
  private _signature: ISignatureDef;
  private _inputs: Node[];
  private _outputs: Node[];
  private _initNodes: Node[];  // Internal init nodes to start initialization.
  private SEPARATOR = ',';
  private _functions: {[key: string]: Graph} = {};
  private _functionExecutorMap: {[key: string]: FunctionExecutor} = {};
  private _resourceManager: ResourceManager;
  private clonedTensorsMap: NamedTensorsMap;
  private keepIntermediateTensors = false;

  get weightIds(): number[] {
    return this.parent ? this.parent.weightIds : this._weightIds;
  }

  get functionExecutorMap(): {[key: string]: FunctionExecutor} {
    return this.parent ? this.parent.functionExecutorMap :
                         this._functionExecutorMap;
  }

  get weightMap(): NamedTensorsMap {
    return this.parent ? this.parent.weightMap : this._weightMap;
  }

  set weightMap(weightMap: NamedTensorsMap) {
    const weightIds = Object.keys(weightMap).map(
        key => weightMap[key].map(tensor => tensor.id));
    this._weightIds = [].concat(...weightIds);
    this._weightMap = weightMap;
  }

  /**
   * Set `ResourceManager` shared by executors of a model.
   * @param resourceManager: `ResourceManager` of the `GraphModel`.
   */
  set resourceManager(resourceManager: ResourceManager) {
    this._resourceManager = resourceManager;
  }

  get inputs(): TensorInfo[] {
    return this._inputs.map(node => {
      return {
        name: node.name,
        shape: node.attrParams['shape'] ?
            node.attrParams['shape'].value as number[] :
            undefined,
        dtype: node.attrParams['dtype'] ?
            node.attrParams['dtype'].value as DataType :
            undefined
      };
    });
  }

  get outputs(): TensorInfo[] {
    return this._outputs.map(node => {
      return {
        name: node.name,
        shape: node.attrParams['shape'] ?
            node.attrParams['shape'].value as number[] :
            undefined,
        dtype: node.attrParams['dtype'] ?
            node.attrParams['dtype'].value as DataType :
            undefined
      };
    });
  }

  get inputNodes(): string[] {
    return this._inputs.map(node => node.signatureKey || node.name);
  }

  get outputNodes(): string[] {
    return this._outputs.map((node) => {
      const name = node.signatureKey || node.name;
      return node.defaultOutput ? (`${name}:${node.defaultOutput}`) : name;
    });
  }

  get functions(): {[key: string]: ISignatureDef} {
    return Object.keys(this._functions).reduce((map, key) => {
      map[key] = this._functions[key].signature;
      return map;
    }, {} as {[key: string]: ISignatureDef});
  }

  /**
   *
   * @param graph Graph the model or function graph to be executed.
   * @param parent When building function exector you need to set the parent
   * executor. Since the weights and function executor maps are set at parant
   * level, that function executor can access the function maps and weight maps
   * through the parent.
   */
  constructor(private graph: Graph, private parent?: GraphExecutor) {
    this._outputs = graph.outputs;
    this._inputs = graph.inputs;
    this._initNodes = graph.initNodes;
    this._signature = graph.signature;
    this._functions = graph.functions;
    // create sub-graph executors
    if (graph.functions != null) {
      Object.keys(graph.functions).forEach(name => {
        this._functionExecutorMap[name] =
            new GraphExecutor(graph.functions[name], this);
      });
    }
  }

  private getCompilationKey(inputs: Node[], outputs: Node[]): string {
    const sortedInputs = inputs.map(node => node.name).sort();
    const sortedOutputs = outputs.map(node => node.name).sort();
    return sortedInputs.join(this.SEPARATOR) + '--' +
        sortedOutputs.join(this.SEPARATOR);
  }

  /**
   * Compiles the inference graph and returns the minimal set of nodes that are
   * required for execution, in the correct execution order.
   * @returns {Object} compilation The compile result.
   * @returns {Node[]} compilation.orderedNodes Nodes in the correct execution
   *     order.
   * @returns {Map<string, Node[]>} compilation.nodeLiveUntilMap A map from node
   *     to disposable nodes after its execution. That is, for a node `x`,
   *     `nodeLiveUntilMap[x]` indicates all nodes whose intermediate
   *     tensors should be disposed after `x` is executed.
   */
  private compile(inputs: NamedTensorMap, outputs: Node[]):
      {orderedNodes: Node[], nodeLiveUntilMap: Map<string, Node[]>} {
    const executionInfo =
        getExecutionSubgraph(inputs, outputs, this.weightMap, this._initNodes);
    const {missingInputs, dynamicNode, syncInputs} = executionInfo;
    if (dynamicNode != null) {
      throw new Error(
          `This execution contains the node '${dynamicNode.name}', which has ` +
          `the dynamic op '${dynamicNode.op}'. Please use ` +
          `model.executeAsync() instead. Alternatively, to avoid the ` +
          `dynamic ops, specify the inputs [${syncInputs}]`);
    }

    if (missingInputs.length > 0) {
      const outNames = outputs.map(n => n.name);
      const inNames = Object.keys(inputs);
      throw new Error(
          `Cannot compute the outputs [${outNames}] from the provided inputs ` +
          `[${inNames}]. Missing the following inputs: [${missingInputs}]`);
    }

    const orderedNodes = getNodesInTopologicalOrder(this.graph, executionInfo);
    const nodeLiveUntilMap = getNodeLiveUntilMap(orderedNodes);
    return {orderedNodes, nodeLiveUntilMap};
  }

  private cloneAndKeepTensor(tensor: Tensor) {
    if (tensor == null) {
      return null;
    }
    const clone = tensor.clone();
    // Keep the clone because`model.execute()` may be called within
    // a `tidy()`, but the user may inspect these tensors after the
    // tidy.
    keep(clone);
    return clone;
  }

  private cloneTensorList(tensors: Tensor[]) {
    if (!tensors) {
      return null;
    }
    const clonedTensor = tensors.map(tensor => {
      return this.cloneAndKeepTensor(tensor);
    });
    return clonedTensor;
  }

  private cloneTensorMap(tensorsMap: NamedTensorsMap): NamedTensorsMap {
    return Object.fromEntries(
        Object.entries(tensorsMap).map(([name, tensorsList]) => {
          return [name, this.cloneTensorList(tensorsList)];
        }));
  }

  /**
   * Executes the inference for given input tensors.
   * @param inputs Tensor map for the model inputs, keyed by the input node
   * names.
   * @param outputs Optional. output node name from the Tensorflow model, if
   * no outputs are specified, the default outputs of the model would be used.
   * You can inspect intermediate nodes of the model by adding them to the
   * outputs array.
   */
  execute(inputs: NamedTensorMap, outputs?: string[]): Tensor[] {
    // Dispose any tensors from a prior run to avoid leaking them.
    this.disposeIntermediateTensors();
    inputs = this.mapInputs(inputs);
    const names = Object.keys(inputs).sort();
    this.checkInputs(inputs);
    this.checkInputShapeAndType(inputs);
    outputs = this.mapOutputs(outputs);
    this.checkOutputs(outputs);
    const inputNodes =
        names.map(name => this.graph.nodes[parseNodeName(name)[0]]);
    const outputNodeNames = outputs.map(name => parseNodeName(name)[0]);
    const outputNodeNameSet = new Set(outputNodeNames);
    let outputNodes = outputNodeNames.map(name => this.graph.nodes[name]);
    // If no outputs are specified, then use the default outputs of the model.
    if (outputNodes.length === 0) {
      outputNodes = this._outputs;
    }

    const compilationKey = this.getCompilationKey(inputNodes, outputNodes);

    // Do nothing if the compiled graph cache contains the input.
    let compilation = this.compiledMap.get(compilationKey);
    if (compilation == null) {
      compilation = this.compile(inputs, outputNodes);
      this.compiledMap.set(compilationKey, compilation);
    }

    // Keep tensors if KEEP_INTERMEDIATE_TENSORS is on.
    try {
      this.keepIntermediateTensors = env().getBool('KEEP_INTERMEDIATE_TENSORS');
    } catch (e) {
      this.keepIntermediateTensors = false;
      console.warn(e.message);
    }
    const tensorArrayMap: TensorArrayMap = {};
    const tensorListMap: TensorListMap = {};

    return tidy(() => {
      const context = new ExecutionContext(
          this.weightMap, tensorArrayMap, tensorListMap,
          this.functionExecutorMap, this.parseNodeNameCache);
      const tensorsMap: NamedTensorsMap = {...this.weightMap};
      if (this.keepIntermediateTensors) {
        this.clonedTensorsMap = this.cloneTensorMap(this.weightMap);
      }

      Object.keys(inputs).forEach(name => {
        const [nodeName, index] = parseNodeName(name, context);
        const tensors: Tensor[] = [];
        tensors[index] = inputs[name];
        tensorsMap[nodeName] = tensors;
        if (this.keepIntermediateTensors) {
          this.clonedTensorsMap[nodeName] = this.cloneTensorList(tensors);
        }
      });

      const tensorsToKeep = this.getFrozenTensorIds(tensorsMap);
      const {orderedNodes, nodeLiveUntilMap} = compilation;
      for (const node of orderedNodes) {
        if (tensorsMap[node.name]) {
          continue;
        }
        const tensors =
            executeOp(node, tensorsMap, context, this._resourceManager) as
            Tensor[];
        if (util.isPromise(tensors)) {
          throw new Error(
              `The execution of the op '${node.op}' returned a promise. ` +
              `Please use model.executeAsync() instead.`);
        }
        tensorsMap[node.name] = tensors;
        if (this.keepIntermediateTensors) {
          this.clonedTensorsMap[node.name] = this.cloneTensorList(tensors);
        }
        this.checkTensorForDisposalWithNodeLiveUntilInfo(
            node, tensorsMap, context, tensorsToKeep, outputNodeNameSet,
            nodeLiveUntilMap.get(node.name));
      }

      // dispose the context for the root executor
      if (this.parent == null) {
        context.dispose(tensorsToKeep);
      }

      return outputs.map(name => getTensor(name, tensorsMap, context));
    });
  }

  private getFrozenTensorIds(tensorMap: NamedTensorsMap): Set<number> {
    const ids = [].concat.apply(
        [],
        Object.keys(tensorMap)
            .map(key => tensorMap[key])
            .map(tensors => tensors.map(tensor => tensor.id)));
    return new Set(ids);
  }

  private checkTensorForDisposal(
      nodeName: string, node: Node, tensorMap: NamedTensorsMap,
      context: ExecutionContext, tensorsToKeep: Set<number>,
      outputNodeNameSet: Set<string>,
      intermediateTensorConsumerCount: {[key: string]: number}) {
    // Skip output nodes and any control flow nodes, since its dependency is
    // tricky to track correctly.
    if (isControlFlow(node) || outputNodeNameSet.has(nodeName)) {
      return;
    }

    for (const tensor of tensorMap[nodeName]) {
      if (tensor == null) {
        continue;
      }
      intermediateTensorConsumerCount[tensor.id] =
          (intermediateTensorConsumerCount[tensor.id] || 0) +
          node.children.length;
    }

    for (const input of node.inputs) {
      // Skip any control flow nodes, since its dependency is tricky to track
      // correctly.
      if (isControlFlow(input)) {
        continue;
      }

      const tensors =
          getTensorsForCurrentContext(input.name, tensorMap, context);
      if (tensors == null) {
        continue;
      }

      for (const tensor of tensors) {
        if (!tensor || tensor.kept || tensorsToKeep.has(tensor.id)) {
          continue;
        }

        // Only intermediate nodes' tensors have counts set, not marked as
        // kept, and not in `tensorsToKeep`.
        // Input and weight nodes' tensors should exist in `tensorsToKeep`.
        // Output and control flow nodes' tensors should never have count set.
        const count = intermediateTensorConsumerCount[tensor.id];
        if (count === 1) {
          tensor.dispose();
          delete intermediateTensorConsumerCount[tensor.id];
        } else if (count != null) {
          intermediateTensorConsumerCount[tensor.id]--;
        }
      }
    }
  }

  private checkTensorForDisposalWithNodeLiveUntilInfo(
      node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
      tensorsToKeep: Set<number>, outputNodeNameSet: Set<string>,
      liveUntilNodes?: Node[]) {
    function isNonDisposableNode(node: Node) {
      // Skip output nodes and any control flow nodes, since its dependency is
      // tricky to track correctly.
      return isControlFlow(node) || outputNodeNameSet.has(node.name);
    }

    if (isControlFlow(node) || liveUntilNodes == null) {
      return;
    }

    for (const nodeToDispose of liveUntilNodes) {
      if (isNonDisposableNode(nodeToDispose)) {
        continue;
      }
      const tensors = getTensorsForCurrentContext(
          nodeToDispose.name, tensorMap, context);
      for (const tensor of tensors) {
        if (!tensor || tensor.kept || tensorsToKeep.has(tensor.id)) {
          continue;
        }
        tensor.dispose();
      }
    }
  }

  /**
   * Executes the inference for given input tensors in Async fashion.
   * @param inputs Tensor map for the model inputs, keyed by the input node
   * names.
   * @param outputs output node name from the Tensorflow model, if no outputs
   * are specified, the default outputs of the model would be used. You can
   * inspect intermediate nodes of the model by adding them to the outputs
   * array.
   */
  async executeAsync(inputs: NamedTensorMap, outputs?: string[]):
      Promise<Tensor[]> {
    return this._executeAsync(inputs, outputs);
  }

  disposeIntermediateTensors() {
    if (!this.clonedTensorsMap) {
      return;
    }
    Object.values(this.clonedTensorsMap).forEach(tensorsList => {
      for (const tensor of tensorsList) {
        if (tensor && !tensor.isDisposed) {
          tensor.dispose();
        }
      }
    });

    this.clonedTensorsMap = null;
  }

  getIntermediateTensors(): NamedTensorsMap {
    return this.clonedTensorsMap;
  }

  /**
   * Executes the inference for given input tensors in Async fashion.
   * @param inputs Tensor map for the model inputs, keyed by the input node
   * names.
   * @param outputs Optional. output node name from the Tensorflow model,
   * if no outputs are specified, the default outputs of the model would be
   * used. You can inspect intermediate nodes of the model by adding them to
   * the outputs array.
   * @param isFunctionExecution Optional. Flag for executing a function.
   * @param tensorArrayMap Optional, global TensorArray map by id. Used for
   * function execution.
   * @param tensorArrayMap Optional global TensorList map by id. Used for
   * function execution.
   */
  private async _executeAsync(
      inputs: NamedTensorMap, outputs?: string[], isFunctionExecution = false,
      tensorArrayMap: TensorArrayMap = {},
      tensorListMap: TensorListMap = {}): Promise<Tensor[]> {
    // Dispose any tensors from a prior run to avoid leaking them.
    this.disposeIntermediateTensors();
    if (!isFunctionExecution) {
      inputs = this.mapInputs(inputs);
      this.checkInputs(inputs);
      this.checkInputShapeAndType(inputs);
      outputs = this.mapOutputs(outputs);
      this.checkOutputs(outputs);
    }

    // Keep tensors if KEEP_INTERMEDIATE_TENSORS is on.
    try {
      this.keepIntermediateTensors = env().getBool('KEEP_INTERMEDIATE_TENSORS');
    } catch (e) {
      this.keepIntermediateTensors = false;
      console.warn(e.message);
    }

    const context = new ExecutionContext(
        this.weightMap, tensorArrayMap, tensorListMap, this.functionExecutorMap,
        this.parseNodeNameCache);

    if (this.keepIntermediateTensors) {
      this.clonedTensorsMap = this.cloneTensorMap(this.weightMap);
    }

    // Graph with control flow op requires runtime evaluation of the execution
    // order, while without control flow the execution order is pre-determined
    // in the compile method.
    const tensorsMap = await this.executeWithControlFlow(
        inputs, context, outputs, isFunctionExecution);
    const results = outputs.map(name => getTensor(name, tensorsMap, context));

    // dispose all the intermediate tensors
    const outputIds = results.map(t => t.id);
    const inputIds = Object.keys(inputs).map(name => inputs[name].id);
    const keepIds =
        new Set<number>([...outputIds, ...inputIds, ...this.weightIds]);

    Object.values(tensorsMap).forEach(tensorsList => {
      tensorsList.forEach(tensor => {
        if (tensor && !tensor.isDisposed && !keepIds.has(tensor.id)) {
          tensor.dispose();
        }
      });
    });

    // dispose the context for the root executor
    if (this.parent == null) {
      context.dispose(keepIds);
    }

    return results;
  }

  async executeFunctionAsync(
      inputs: Tensor[], tensorArrayMap: TensorArrayMap,
      tensorListMap: TensorListMap): Promise<Tensor[]> {
    const mappedInputs = inputs.reduce((map, tensor, index) => {
      map[this.inputs[index].name] = tensor;
      return map;
    }, {} as NamedTensorMap);

    return this._executeAsync(
        mappedInputs, this.outputNodes, true, tensorArrayMap, tensorListMap);
  }

  /**
   * When there are control flow nodes in the graph, the graph execution use
   * ExecutionContext to keep track of the frames and loop iterators.
   * @param inputs placeholder tensors for the graph.
   * @param context the execution context object for current execution.
   * @param outputNames Optional. output node name from the Tensorflow model,
   * if no outputs are specified, the default outputs of the model would be
   * used. You can inspect intermediate nodes of the model by adding them to
   * the outputs array.
   * @param isFunctionExecution Flag for executing a function.
   */
  private async executeWithControlFlow(
      inputs: NamedTensorMap, context: ExecutionContext, outputNames?: string[],
      isFunctionExecution?: boolean): Promise<NamedTensorsMap> {
    const names = Object.keys(inputs);
    const inputNodes =
        names.map(name => this.graph.nodes[parseNodeName(name)[0]]);
    const outputNodeNames = outputNames.map(name => parseNodeName(name)[0]);
    const outputNodeNameSet = new Set(outputNodeNames);
    let outputNodes = outputNodeNames.map(name => this.graph.nodes[name]);

    // If no outputs are specified, then use the default outputs of the model.
    if (outputNodes.length === 0) {
      outputNodes = this._outputs;
    }

    const {usedNodes, missingInputs, dynamicNode, syncInputs} =
        getExecutionSubgraph(
            inputs, outputNodes, this.weightMap, this._initNodes);

    // First nodes to execute include inputNodes, weights, and initNodes.
    const stack: NodeWithContexts[] = [
      ...inputNodes, ...this.graph.weights, ...(this._initNodes || [])
    ].map(node => {
      return {node, contexts: context.currentContext};
    });
    const tensorsMap: NamedTensorsMap = {...this.weightMap};
    Object.keys(inputs).forEach(name => {
      const [nodeName, index] = parseNodeName(name);
      const tensors: Tensor[] = [];
      tensors[index] = inputs[name];
      tensorsMap[nodeName] = tensors;
    });
    const intermediateTensorConsumerCount: {[key: number]: number} = {};
    const tensorsToKeep = this.getFrozenTensorIds(tensorsMap);
    const added: {[key: string]: boolean} = {};
    while (stack.length > 0) {
      const promises = this.processStack(
          inputNodes, stack, context, tensorsMap, added, tensorsToKeep,
          outputNodeNameSet, intermediateTensorConsumerCount, usedNodes);
      await Promise.all(promises);
    }
    if (dynamicNode == null && !isFunctionExecution) {
      console.warn(
          `This model execution did not contain any nodes with control flow ` +
          `or dynamic output shapes. You can use model.execute() instead.`);
    }
    const missingOutputs =
        outputNodes
            .filter(
                node => !isControlFlow(node) &&
                    !getTensor(node.name, tensorsMap, context))
            .map(node => node.name);
    if (missingOutputs.length > 0) {
      let alternativeMsg = '';
      if (dynamicNode != null) {
        alternativeMsg =
            `Alternatively, to avoid the dynamic ops, use model.execute() ` +
            `and specify the inputs [${syncInputs}]`;
      }
      throw new Error(
          `Cannot compute the outputs [${missingOutputs}] from the provided ` +
          `inputs [${names}]. Consider providing the following inputs: ` +
          `[${missingInputs}]. ${alternativeMsg}`);
    }
    return tensorsMap;
  }

  private processStack(
      inputNodes: Node[], stack: NodeWithContexts[], context: ExecutionContext,
      tensorMap: NamedTensorsMap, added: {[key: string]: boolean},
      tensorsToKeep: Set<number>, outputNodeNameSet: Set<string>,
      intermediateTensorConsumerCount: {[key: number]: number},
      usedNodes: Set<string>) {
    const promises: Array<Promise<Tensor[]>> = [];
    while (stack.length > 0) {
      const item = stack.pop();
      context.currentContext = item.contexts;
      let nodeName = '';
      // The tensor of the Enter op with isConstant set should be set
      // in the parent scope, so it will be available as constant for the
      // whole loop.
      if (item.node.op === 'Enter' &&
          getParamValue('isConstant', item.node, tensorMap, context)) {
        [nodeName] = getNodeNameAndIndex(item.node.name, context);
      }

      // only process nodes that are not in the tensorMap yet, this include
      // inputNodes and internal initNodes.
      if (tensorMap[item.node.name] == null) {
        const tensors =
            executeOp(item.node, tensorMap, context, this._resourceManager);
        if (!nodeName) {
          [nodeName] = getNodeNameAndIndex(item.node.name, context);
        }
        const currentContext = context.currentContext;
        if (util.isPromise(tensors)) {
          promises.push(tensors.then(t => {
            tensorMap[nodeName] = t;
            if (this.keepIntermediateTensors) {
              this.clonedTensorsMap[nodeName] = this.cloneTensorList(t);
            }
            context.currentContext = currentContext;
            this.checkTensorForDisposal(
                nodeName, item.node, tensorMap, context, tensorsToKeep,
                outputNodeNameSet, intermediateTensorConsumerCount);
            this.processChildNodes(
                item.node, stack, context, tensorMap, added, usedNodes);
            return t;
          }));
        } else {
          tensorMap[nodeName] = tensors;
          if (this.keepIntermediateTensors) {
            this.clonedTensorsMap[nodeName] = this.cloneTensorList(tensors);
          }
          this.checkTensorForDisposal(
              nodeName, item.node, tensorMap, context, tensorsToKeep,
              outputNodeNameSet, intermediateTensorConsumerCount);
          this.processChildNodes(
              item.node, stack, context, tensorMap, added, usedNodes);
        }
      } else {
        this.processChildNodes(
            item.node, stack, context, tensorMap, added, usedNodes);
      }
    }
    return promises;
  }

  private processChildNodes(
      node: Node, stack: NodeWithContexts[], context: ExecutionContext,
      tensorMap: NamedTensorsMap, added: {[key: string]: boolean},
      usedNodes: Set<string>) {
    node.children.forEach((childNode) => {
      const [nodeName, ] = getNodeNameAndIndex(childNode.name, context);
      if (added[nodeName] || !usedNodes.has(childNode.name)) {
        return;
      }
      // Merge op can be pushed if any of its inputs has value.
      if (childNode.op === 'Merge') {
        if (childNode.inputNames.some(name => {
              return !!getTensor(name, tensorMap, context);
            })) {
          added[nodeName] = true;
          stack.push({contexts: context.currentContext, node: childNode});
        }
      } else  // Otherwise all inputs must to have value.
          if (childNode.inputNames.every(name => {
                return !!getTensor(name, tensorMap, context);
              })) {
        added[nodeName] = true;
        stack.push({contexts: context.currentContext, node: childNode});
      }
    });
  }

  /**
   * Releases the memory used by the weight tensors.
   */
  dispose() {
    Object.keys(this.weightMap)
        .forEach(
            key => this.weightMap[key].forEach(tensor => tensor.dispose()));
  }

  private checkInputShapeAndType(inputs: NamedTensorMap) {
    Object.keys(inputs).forEach(name => {
      const input = inputs[name];
      const [nodeName, ] = parseNodeName(name);
      const node = this.graph.nodes[nodeName];
      if (node.attrParams['shape'] && node.attrParams['shape'].value) {
        const shape = node.attrParams['shape'].value as number[];
        const match = shape.length === input.shape.length &&
            input.shape.every(
                (dim, index) => shape[index] === -1 || shape[index] === dim);
        util.assert(
            match,
            () => `The shape of dict['${node.name}'] provided in ` +
                `model.execute(dict) must be [${shape}], but was ` +
                `[${input.shape}]`);
      }
      if (node.attrParams['dtype'] && node.attrParams['dtype'].value) {
        util.assert(
            input.dtype === node.attrParams['dtype'].value as string,
            () => `The dtype of dict['${node.name}'] provided in ` +
                `model.execute(dict) must be ` +
                `${node.attrParams['dtype'].value}, but was ${input.dtype}`);
      }
    });
  }

  private mapInputs(inputs: NamedTensorMap) {
    const result: NamedTensorMap = {};
    for (const inputName in inputs) {
      const tensor = this._signature ?.inputs ?.[inputName];
      if (tensor != null) {
        result[tensor.name] = inputs[inputName];
      } else {
        result[inputName] = inputs[inputName];
      }
    }
    return result;
  }

  private checkInputs(inputs: NamedTensorMap) {
    const notInGraph = Object.keys(inputs).filter(name => {
      const [nodeName] = parseNodeName(name);
      return this.graph.nodes[nodeName] == null;
    });
    if (notInGraph.length > 0) {
      throw new Error(
          `The dict provided in model.execute(dict) has ` +
          `keys: [${notInGraph}] that are not part of graph`);
    }
  }

  private mapOutputs(outputs: string[]) {
    return outputs.map(name => {
      const tensor = this._signature ?.outputs ?.[name];
      if (tensor != null) {
        return tensor.name;
      }
      return name;
    }, {});
  }

  private checkOutputs(outputs: string[]): void {
    outputs.forEach(name => {
      const [normalizedName] = parseNodeName(name);
      if (!this.graph.nodes[normalizedName]) {
        throw new Error(`The output '${name}' is not found in the graph`);
      }
    });
  }
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {concat, DataType, keep, reshape, scalar, slice, stack, Tensor, tensor, tidy, unstack} from '@tensorflow/tfjs-core';

import {assertShapesMatchAllowUndefinedSize} from './tensor_utils';

export interface TensorWithState {
  tensor?: Tensor;
  written?: boolean;
  read?: boolean;
  cleared?: boolean;
}
/**
 * The TensorArray object keeps an array of Tensors.  It
 * allows reading from the array and writing to the array.
 */
export class TensorArray {
  private tensors: TensorWithState[] = [];
  private closed_ = false;
  readonly idTensor: Tensor;
  constructor(
      readonly name: string, readonly dtype: DataType, private maxSize: number,
      private elementShape: number[], readonly identicalElementShapes: boolean,
      readonly dynamicSize: boolean, readonly clearAfterRead: boolean) {
    this.idTensor = scalar(0);
    keep(this.idTensor);
  }

  get id() {
    return this.idTensor.id;
  }

  get closed() {
    return this.closed_;
  }

  /**
   * Dispose the tensors and idTensor and mark the TensoryArray as closed.
   */
  clearAndClose(keepIds?: Set<number>) {
    this.tensors.forEach(tensor => {
      if (keepIds == null || !keepIds.has(tensor.tensor.id)) {
        tensor.tensor.dispose();
      }
    });
    this.tensors = [];
    this.closed_ = true;
    this.idTensor.dispose();
  }

  size(): number {
    return this.tensors.length;
  }

  /**
   * Read the value at location index in the TensorArray.
   * @param index Number the index to read from.
   */
  read(index: number): Tensor {
    if (this.closed_) {
      throw new Error(`TensorArray ${this.name} has already been closed.`);
    }

    if (index < 0 || index >= this.size()) {
      throw new Error(`Tried to read from index ${index}, but array size is: ${
          this.size()}`);
    }

    const tensorWithState = this.tensors[index];
    if (tensorWithState.cleared) {
      throw new Error(
          `TensorArray ${this.name}: Could not read index ${
              index} twice because it was cleared after a previous read ` +
          `(perhaps try setting clear_after_read = false?).`);
    }

    if (this.clearAfterRead) {
      tensorWithState.cleared = true;
    }

    tensorWithState.read = true;
    return tensorWithState.tensor;
  }

  /**
   * Helper method to read multiple tensors from the specified indices.
   */
  readMany(indices: number[]): Tensor[] {
    return indices.map(index => this.read(index));
  }

  /**
   * Write value into the index of the TensorArray.
   * @param index number the index to write to.
   * @param tensor
   */
  write(index: number, tensor: Tensor) {
    if (this.closed_) {
      throw new Error(`TensorArray ${this.name} has already been closed.`);
    }

    if (index < 0 || !this.dynamicSize && index >= this.maxSize) {
      throw new Error(`Tried to write to index ${
          index}, but array is not resizeable and size is: ${this.maxSize}`);
    }

    const t = this.tensors[index] || {};

    if (tensor.dtype !== this.dtype) {
      throw new Error(`TensorArray ${
          this.name}: Could not write to TensorArray index ${index},
          because the value dtype is ${
          tensor.dtype}, but TensorArray dtype is ${this.dtype}.`);
    }

    // Set the shape for the first time write to unknow shape tensor array
    if (this.size() === 0 &&
        (this.elementShape == null || this.elementShape.length === 0)) {
      this.elementShape = tensor.shape;
    }

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, tensor.shape,
        `TensorArray ${this.name}: Could not write to TensorArray index ${
            index}.`);

    if (t.read) {
      throw new Error(
          `TensorArray ${this.name}: Could not write to TensorArray index ${
              index}, because it has already been read.`);
    }

    if (t.written) {
      throw new Error(
          `TensorArray ${this.name}: Could not write to TensorArray index ${
              index}, because it has already been written.`);
    }

    t.tensor = tensor;
    keep(tensor);
    t.written = true;

    this.tensors[index] = t;
  }

  /**
   * Helper method to write multiple tensors to the specified indices.
   */
  writeMany(indices: number[], tensors: Tensor[]) {
    if (indices.length !== tensors.length) {
      throw new Error(
          `TensorArray ${this.name}: could not write multiple tensors,` +
          `because the index size: ${
              indices.length} is not the same as tensors size: ${
              tensors.length}.`);
    }

    indices.forEach((i, index) => this.write(i, tensors[index]));
  }

  /**
   * Return selected values in the TensorArray as a packed Tensor. All of
   * selected values must have been written and their shapes must all match.
   * @param [indices] number[] Optional. Taking values in [0, max_value). If the
   *    TensorArray is not dynamic, max_value=size(). If not specified returns
   *    all tensors in the original order.
   * @param [dtype]
   */
  gather(indices?: number[], dtype?: DataType): Tensor {
    if (!!dtype && dtype !== this.dtype) {
      throw new Error(`TensorArray dtype is ${
          this.dtype} but gather requested dtype ${dtype}`);
    }

    if (!indices) {
      indices = [];
      for (let i = 0; i < this.size(); i++) {
        indices.push(i);
      }
    } else {
      indices = indices.slice(0, this.size());
    }

    if (indices.length === 0) {
      return tensor([], [0].concat(this.elementShape));
    }

    // Read all the PersistentTensors into a vector to keep track of
    // their memory.
    const tensors = this.readMany(indices);

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, tensors[0].shape, 'TensorArray shape mismatch: ');

    return stack(tensors, 0);
  }

  /**
   * Return the values in the TensorArray as a concatenated Tensor.
   */
  concat(dtype?: DataType): Tensor {
    if (!!dtype && dtype !== this.dtype) {
      throw new Error(`TensorArray dtype is ${
          this.dtype} but concat requested dtype ${dtype}`);
    }

    if (this.size() === 0) {
      return tensor([], [0].concat(this.elementShape));
    }

    const indices = [];
    for (let i = 0; i < this.size(); i++) {
      indices.push(i);
    }
    // Collect all the tensors from the tensors array.
    const tensors = this.readMany(indices);

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, tensors[0].shape,
        `TensorArray shape mismatch: tensor array shape (${
            this.elementShape}) vs first tensor shape (${tensors[0].shape})`);

    return concat(tensors, 0);
  }

  /**
   * Scatter the values of a Tensor in specific indices of a TensorArray.
   * @param indices number[] values in [0, max_value). If the
   *    TensorArray is not dynamic, max_value=size().
   * @param tensor Tensor input tensor.
   */
  scatter(indices: number[], tensor: Tensor) {
    if (tensor.dtype !== this.dtype) {
      throw new Error(`TensorArray dtype is ${
          this.dtype} but tensor has dtype ${tensor.dtype}`);
    }

    if (indices.length !== tensor.shape[0]) {
      throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${
          indices.length} vs. ${tensor.shape[0]}`);
    }

    const maxIndex = Math.max(...indices);

    if (!this.dynamicSize && maxIndex >= this.maxSize) {
      throw new Error(
          `Max index must be < array size (${maxIndex}  vs. ${this.maxSize})`);
    }

    this.writeMany(indices, unstack(tensor, 0));
  }

  /**
   * Split the values of a Tensor into the TensorArray.
   * @param length number[] with the lengths to use when splitting value along
   *    its first dimension.
   * @param tensor Tensor, the tensor to split.
   */
  split(length: number[], tensor: Tensor) {
    if (tensor.dtype !== this.dtype) {
      throw new Error(`TensorArray dtype is ${
          this.dtype} but tensor has dtype ${tensor.dtype}`);
    }
    let totalLength = 0;
    const cumulativeLengths = length.map(len => {
      totalLength += len;
      return totalLength;
    });

    if (totalLength !== tensor.shape[0]) {
      throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${totalLength}, and tensor's shape is: ${tensor.shape}`);
    }

    if (!this.dynamicSize && length.length !== this.maxSize) {
      throw new Error(
          `TensorArray's size is not equal to the size of lengths (${
              this.maxSize} vs. ${length.length}), ` +
          'and the TensorArray is not marked as dynamically resizeable');
    }

    const elementPerRow = totalLength === 0 ? 0 : tensor.size / totalLength;
    const tensors: Tensor[] = [];
    tidy(() => {
      tensor = reshape(tensor, [1, totalLength, elementPerRow]);
      for (let i = 0; i < length.length; ++i) {
        const previousLength = (i === 0) ? 0 : cumulativeLengths[i - 1];
        const indices = [0, previousLength, 0];
        const sizes = [1, length[i], elementPerRow];
        tensors[i] = reshape(slice(tensor, indices, sizes), this.elementShape);
      }
      return tensors;
    });
    const indices = [];
    for (let i = 0; i < length.length; i++) {
      indices[i] = i;
    }
    this.writeMany(indices, tensors);
  }
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import {Tensor} from '@tensorflow/tfjs-core';

import {NamedTensorsMap, TensorArrayMap, TensorListMap} from '../data/types';

import {TensorArray} from './tensor_array';
import {TensorList} from './tensor_list';
import {FunctionExecutor} from './types';

export interface ExecutionContextInfo {
  id: number;           // the unique id of the context info
  frameName: string;    // The frame name of the loop, this comes from
                        // the TensorFlow NodeDef.
  iterationId: number;  // The iteration id of the loop
}

/**
 * ExecutionContext captures the runtime environment of the node. It keeps
 * track of the current frame and iteration for the control flow ops.
 *
 * For example, typical Dynamic RNN model may contain loops, for which
 * TensorFlow will generate graphs with Enter/Exit nodes to control the
 * current execution frame, and NextIteration Nodes for iteration id increment.
 * For model with branch logic, TensorFLow will generate Switch/Merge ops.
 */
export class ExecutionContext {
  private rootContext = {id: 0, frameName: '', iterationId: 0};
  private contexts: ExecutionContextInfo[] = [this.rootContext];
  private lastId = 0;
  private _currentContextIds: string[];

  constructor(
      readonly weightMap: NamedTensorsMap = {},
      readonly tensorArrayMap: TensorArrayMap = {},
      readonly tensorListMap: TensorListMap = {},
      readonly functionMap: {[key: string]: FunctionExecutor} = {},
      readonly parseNodeNameCache?: Map<string, [string, number, string?]>) {
    this.generateCurrentContextIds();
  }

  private newFrame(id: number, frameName: string) {
    return {id, frameName, iterationId: 0};
  }

  /**
   * Set the current context
   * @param contexts: ExecutionContextInfo[] the current path of execution
   * frames
   */
  set currentContext(contexts: ExecutionContextInfo[]) {
    if (this.contexts !== contexts) {
      this.contexts = contexts;
      this.generateCurrentContextIds();
    }
  }

  get currentContext(): ExecutionContextInfo[] {
    return this.contexts;
  }

  /**
   * Returns the current context in string format.
   */
  get currentContextId(): string {
    return this._currentContextIds[0];
  }

  /**
   * Returns the current context and all parent contexts in string format.
   * This allow access to the nodes in the current and parent frames.
   */
  get currentContextIds(): string[] {
    return this._currentContextIds;
  }

  private generateCurrentContextIds() {
    const names = [];
    for (let i = 0; i < this.contexts.length - 1; i++) {
      const contexts = this.contexts.slice(0, this.contexts.length - i);
      names.push(this.contextIdforContexts(contexts));
    }
    names.push('');
    this._currentContextIds = names;
  }

  private contextIdforContexts(contexts: ExecutionContextInfo[]) {
    return contexts ?
        contexts
            .map(
                context => (context.id === 0 && context.iterationId === 0) ?
                    '' :
                    `${context.frameName}-${context.iterationId}`)
            .join('/') :
        '';
  }

  /**
   * Enter a new frame, a new context is pushed on the current context list.
   * @param frameId new frame id
   */
  enterFrame(frameId: string) {
    if (this.contexts) {
      this.lastId++;
      this.contexts = this.contexts.slice();
      this.contexts.push(this.newFrame(this.lastId, frameId));
      this._currentContextIds.unshift(this.contextIdforContexts(this.contexts));
    }
  }

  /**
   * Exit the current frame, the last context is removed from the current
   * context list.
   */
  exitFrame() {
    if (this.contexts && this.contexts.length > 1) {
      this.contexts = this.contexts.slice();
      this.contexts.splice(-1);
      this.currentContextIds.shift();
    } else {
      throw new Error('Cannot exit frame, the context is empty');
    }
  }

  /**
   * Enter the next iteration of a loop, the iteration id of last context is
   * increased.
   */
  nextIteration() {
    if (this.contexts && this.contexts.length > 0) {
      this.contexts = this.contexts.slice();
      this.lastId++;
      const context =
          Object.assign({}, this.contexts[this.contexts.length - 1]);
      context.iterationId += 1;
      context.id = this.lastId;
      this.contexts.splice(-1, 1, context);
      this._currentContextIds.splice(
          0, 1, this.contextIdforContexts(this.contexts));
    } else {
      throw new Error('Cannot increase frame iteration, the context is empty');
    }
  }

  getWeight(name: string): Tensor[] {
    return this.weightMap[name];
  }

  addTensorArray(tensorArray: TensorArray) {
    this.tensorArrayMap[tensorArray.id] = tensorArray;
  }

  getTensorArray(id: number): TensorArray {
    return this.tensorArrayMap[id];
  }

  addTensorList(tensorList: TensorList) {
    this.tensorListMap[tensorList.id] = tensorList;
  }

  getTensorList(id: number): TensorList {
    return this.tensorListMap[id];
  }

  dispose(keepIds: Set<number>) {
    for (const key in this.tensorArrayMap) {
      this.tensorArrayMap[key].clearAndClose(keepIds);
    }

    for (const key in this.tensorListMap) {
      this.tensorListMap[key].clearAndClose(keepIds);
    }
  }
}

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {NamedTensorMap} from '@tensorflow/tfjs-core';

import {NamedTensorsMap} from '../data/types';
import {parseNodeName} from '../operations/executors/utils';
import {Graph, Node} from '../operations/types';

export interface ExecutionInfo {
  inputs: NamedTensorMap;
  outputs: Node[];
  usedNodes: Set<string>;
  missingInputs: string[];
  dynamicNode: Node;
  syncInputs: string[];
}

/**
 * Given graph inputs and desired outputs, find the minimal set of nodes
 * to execute in order to compute the outputs. In addition return other useful
 * info such:
 * - Missing inputs needed to compute the output.
 * - Whether the subgraph contains dynamic ops (control flow, dynamic shape).
 * - Alternative inputs in order to avoid async (dynamic op) execution.
 */
export function getExecutionSubgraph(
    inputs: NamedTensorMap, outputs: Node[], weightMap: NamedTensorsMap,
    initNodes?: Node[]): ExecutionInfo {
  const usedNodes = new Set<string>();
  const missingInputs: string[] = [];
  let dynamicNode: Node = null;
  let syncInputs: string[] = null;

  // Start with the outputs, going backwards and find all the nodes that are
  // needed to compute those outputs.
  const seen = new Set<string>();
  const inputNodeNames =
      new Set(Object.keys(inputs).map((name) => parseNodeName(name)[0]));

  initNodes = initNodes || [];
  const initNodeNames =
      new Set(initNodes.map((node) => parseNodeName(node.name)[0]));

  const frontier = [...outputs];
  while (frontier.length > 0) {
    const node = frontier.pop();
    if (isControlFlow(node) || isDynamicShape(node) || isHashTable(node)) {
      if (dynamicNode == null) {
        dynamicNode = node;
        syncInputs = dynamicNode.children.map(child => child.name)
                         .filter(name => usedNodes.has(name));
      }
    }
    usedNodes.add(node.name);

    // Weights are dead end since we already have their values.
    if (weightMap[node.name] != null) {
      continue;
    }
    // This node is a dead end since it's one of the user-provided inputs.
    if (inputNodeNames.has(node.name)) {
      continue;
    }
    // This node is a dead end since it doesn't have any inputs.
    if (initNodeNames.has(node.name)) {
      continue;
    }
    if (node.inputs.length === 0) {
      missingInputs.push(node.name);
      continue;
    }
    node.inputs.forEach(input => {
      // Don't add to the frontier if it is already there.
      if (seen.has(input.name)) {
        return;
      }
      seen.add(input.name);
      frontier.push(input);
    });
  }
  return {inputs, outputs, usedNodes, missingInputs, dynamicNode, syncInputs};
}

/**
 * Given the execution info, return a list of nodes in topological order that
 * need to be executed to compute the output.
 */
export function getNodesInTopologicalOrder(
    graph: Graph, executionInfo: ExecutionInfo): Node[] {
  const {usedNodes, inputs} = executionInfo;
  const inputNodes = Object.keys(inputs)
                         .map(name => parseNodeName(name)[0])
                         .map(name => graph.nodes[name]);
  const initNodes = graph.initNodes || [];

  const isUsed = (node: Node|string) =>
      usedNodes.has(typeof node === 'string' ? node : node.name);

  function unique(nodes: Node[]): Node[] {
    return [...new Map(nodes.map((node) => [node.name, node])).values()];
  }
  const predefinedNodes = unique([
                            ...inputNodes,
                            ...graph.weights,
                            ...initNodes,
                          ]).filter(isUsed);
  const allNodes = unique([
                     ...predefinedNodes,
                     ...Object.values(graph.nodes),
                   ]).filter(isUsed);
  const nameToNode =
      new Map<string, Node>(allNodes.map((node) => [node.name, node]));

  const inCounts: Record<string, number> = {};
  for (const node of allNodes) {
    inCounts[node.name] = inCounts[node.name] || 0;
    for (const child of node.children) {
      // When the child is unused, set in counts to infinity so that it will
      // never be decreased to 0 and added to the execution list.
      if (!isUsed(child)) {
        inCounts[child.name] = Number.POSITIVE_INFINITY;
      }
      inCounts[child.name] = (inCounts[child.name] || 0) + 1;
    }
  }

  // Build execution order for all used nodes regardless whether they are
  // predefined or not.
  const frontier = Object.entries(inCounts)
                       .filter(([, inCount]) => inCount === 0)
                       .map(([name]) => name);
  const orderedNodeNames = [...frontier];
  while (frontier.length > 0) {
    const nodeName = frontier.pop();
    const node = nameToNode.get(nodeName)!;
    for (const child of node.children.filter(isUsed)) {
      if (--inCounts[child.name] === 0) {
        orderedNodeNames.push(child.name);
        frontier.push(child.name);
      }
    }
  }

  const orderedNodes = orderedNodeNames.map((name) => nameToNode.get(name));
  const filteredOrderedNodes =
      filterPredefinedReachableNodes(orderedNodes, predefinedNodes);

  // TODO: Turn validation on/off with tf env flag.
  validateNodesExecutionOrder(filteredOrderedNodes, predefinedNodes);

  return filteredOrderedNodes;
}

/**
 * This is a helper function of `getNodesInTopologicalOrder`.
 * Returns ordered nodes reachable by at least one predefined node.
 * This can help us filter out redundant nodes from the returned node list.
 * For example:
 * If we have four nodes with dependencies like this:
 *   a --> b --> c --> d
 * when node `c` is predefined (e.g. given as an input tensor), we can
 * skip node `a` and `b` since their outputs will never be used.
 *
 * @param orderedNodes Graph nodes in execution order.
 * @param predefinedNodes Graph inputs, weights, and init nodes. Nodes in this
 *     list must have distinct names.
 */
function filterPredefinedReachableNodes(
    orderedNodes: Node[], predefinedNodes: Node[]) {
  const nameToNode =
      new Map<string, Node>(orderedNodes.map((node) => [node.name, node]));

  // TODO: Filter out more nodes when >=2 nodes are predefined in a path.
  const stack = predefinedNodes.map((node) => node.name);
  const predefinedReachableNodeNames = new Set(stack);
  // Perform a DFS starting from the set of all predefined nodes
  // to find the set of all nodes reachable from the predefined nodes.
  while (stack.length > 0) {
    const nodeName = stack.pop();
    const node = nameToNode.get(nodeName)!;
    for (const child of node.children) {
      if (!nameToNode.has(child.name) ||
          predefinedReachableNodeNames.has(child.name)) {
        continue;
      }
      predefinedReachableNodeNames.add(child.name);
      stack.push(child.name);
    }
  }

  // Filter out unreachable nodes and build the ordered node list.
  const filteredOrderedNodes = orderedNodes.filter(
      (node) => predefinedReachableNodeNames.has(node.name));

  return filteredOrderedNodes;
}

class NodesExecutionOrderError extends Error {
  constructor(message: string) {
    super(`NodesExecutionOrderError: ${message}`);
  }
}

/**
 * This is a helper function of `getNodesInTopologicalOrder`.
 * Validates property: given nodes `a` and `b`, Order(a) > Order(b) if `a`
 * is a child of `b`. This function throws an error if validation fails.
 *
 * @param orderedNodes Graph nodes in execution order.
 * @param predefinedNodes Graph inputs, weights, and init nodes. Nodes in this
 *     list must have distinct names.
 */
function validateNodesExecutionOrder(
    orderedNodes: Node[], predefinedNodes: Node[]) {
  const nodeNameToOrder = new Map<string, number>(
      orderedNodes.map((node, order) => [node.name, order]));
  const predefinedNodeNames = new Set(predefinedNodes.map((node) => node.name));
  const isPredefined = (node: Node|string) =>
      predefinedNodeNames.has(typeof node === 'string' ? node : node.name);
  const willBeExecutedNodeNames =
      new Set(orderedNodes.map((node) => node.name));
  const willBeExecuted = (node: Node|string) =>
      willBeExecutedNodeNames.has(typeof node === 'string' ? node : node.name);

  for (const node of orderedNodes) {
    for (const child of node.children.filter(willBeExecuted)) {
      if (!nodeNameToOrder.has(child.name)) {
        throw new NodesExecutionOrderError(
            `Child ${child.name} of node ${node.name} is unreachable.`);
      }
      if (nodeNameToOrder.get(node.name) > nodeNameToOrder.get(child.name)) {
        throw new NodesExecutionOrderError(`Node ${
            node.name} is scheduled to run after its child ${child.name}.`);
      }
    }
    if (!isPredefined(node)) {
      for (const input of node.inputs) {
        if (!nodeNameToOrder.has(input.name)) {
          throw new NodesExecutionOrderError(
              `Input ${input.name} of node ${node.name} is unreachable.`);
        }
        if (nodeNameToOrder.get(input.name) > nodeNameToOrder.get(node.name)) {
          throw new NodesExecutionOrderError(`Node ${
              node.name} is scheduled to run before its input ${input.name}.`);
        }
      }
    }
  }
}

/**
 * Given the execution info, return a map from node name to the disposable
 * node name list after its execution.
 *
 * @returns A map from node name to disposable nodes after its
 *     execution. That is, for a node `x`, `nodeLiveUntilMap[x]` indicates
 *     all nodes which their intermediate tensors should be disposed after `x`
 *     being executed.
 */
export function getNodeLiveUntilMap(orderedNodes: Node[]): Map<string, Node[]> {
  const nodeNameToOrder = new Map<string, number>(
      orderedNodes.map((node, order) => [node.name, order]));

  const INF_LIFE = Number.MAX_SAFE_INTEGER;
  // Make control flow nodes (and consequently their direct parents)
  // live forever since they're tricky to track correctly.
  const selfLifespans = orderedNodes.map(
      (node, nodeOrder) => isControlFlow(node) ? INF_LIFE : nodeOrder);
  const getSelfLifeSpan = (node: Node) => {
    const selfLife = selfLifespans[nodeNameToOrder.get(node.name)!];
    if (selfLife == null) {
      // If nodeToOrder does not contain the node, it is unused or
      // unreachable in graph.
      return -1;
    }
    return selfLife;
  };

  // `liveUntil[i]` points to the last node in the `orderedNodes` array that
  // may depend on tensors from node `i`. It indicates that all the
  // intermediate tensors from `orderedNodes[i]` should be disposed after
  // `orderedNodes[liveUntil[i]]` is executed.
  // A node lives long enough to pass on its tensors to its children.
  // It lives until at least `max(node's position, children's positions)`.
  const liveUntilOrders = orderedNodes.map((node, nodeOrder) => {
    return node.children.map(getSelfLifeSpan)
        .reduce((a, b) => Math.max(a, b), selfLifespans[nodeOrder]);
  });

  // liveUntilMap:
  // - Key: Name of a node `x`
  // - Values: All nodes whose intermediate tensors should be disposed
  //           after `x` is executed.
  const liveUntilMap = new Map<string, Node[]>();
  for (let nodeOrder = 0; nodeOrder < orderedNodes.length; ++nodeOrder) {
    const liveUntilOrder = liveUntilOrders[nodeOrder];
    if (liveUntilOrder === INF_LIFE) {
      continue;
    }
    const node = orderedNodes[nodeOrder];
    const liveUntilNode = orderedNodes[liveUntilOrder];
    if (!liveUntilMap.has(liveUntilNode.name)) {
      liveUntilMap.set(liveUntilNode.name, []);
    }
    liveUntilMap.get(liveUntilNode.name)!.push(node);
  }
  return liveUntilMap;
}

const CONTROL_FLOW_OPS = new Set([
  'Switch', 'Merge', 'Enter', 'Exit', 'NextIteration', 'StatelessIf',
  'StatelessWhile', 'if', 'While'
]);
const DYNAMIC_SHAPE_OPS = new Set([
  'NonMaxSuppressionV2', 'NonMaxSuppressionV3', 'NonMaxSuppressionV5', 'Where'
]);
const HASH_TABLE_OPS = new Set([
  'HashTable', 'HashTableV2', 'LookupTableImport', 'LookupTableImportV2',
  'LookupTableFind', 'LookupTableFindV2', 'LookupTableSize', 'LookupTableSizeV2'
]);

export function isControlFlow(node: Node) {
  return CONTROL_FLOW_OPS.has(node.op);
}

export function isDynamicShape(node: Node) {
  return DYNAMIC_SHAPE_OPS.has(node.op);
}

export function isHashTable(node: Node) {
  return HASH_TABLE_OPS.has(node.op);
}


/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * This differs from util.assertShapesMatch in that it allows values of
 * negative one, an undefined size of a dimensinon, in a shape to match
 * anything.
 */

import {Tensor, util} from '@tensorflow/tfjs-core';

/**
 * Used by TensorList and TensorArray to verify if elementShape matches, support
 * negative value as the dim shape.
 * @param shapeA
 * @param shapeB
 * @param errorMessagePrefix
 */
export function assertShapesMatchAllowUndefinedSize(
    shapeA: number|number[], shapeB: number|number[],
    errorMessagePrefix = ''): void {
  // constant shape means unknown rank
  if (typeof shapeA === 'number' || typeof shapeB === 'number') {
    return;
  }
  util.assert(
      shapeA.length === shapeB.length,
      () => errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
  for (let i = 0; i < shapeA.length; i++) {
    const dim0 = shapeA[i];
    const dim1 = shapeB[i];
    util.assert(
        dim0 < 0 || dim1 < 0 || dim0 === dim1,
        () =>
            errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
  }
}

export function fullDefinedShape(elementShape: number|number[]): boolean {
  if (typeof elementShape === 'number' || elementShape.some(dim => dim < 0)) {
    return false;
  }
  return true;
}
/**
 * Generate the output element shape from the list elementShape, list tensors
 * and input param.
 * @param listElementShape
 * @param tensors
 * @param elementShape
 */
export function inferElementShape(
    listElementShape: number|number[], tensors: Tensor[],
    elementShape: number|number[]): number[] {
  let partialShape = mergeElementShape(listElementShape, elementShape);
  const notfullDefinedShape = !fullDefinedShape(partialShape);
  if (notfullDefinedShape && tensors.length === 0) {
    throw new Error(
        `Tried to calculate elements of an empty list` +
        ` with non-fully-defined elementShape: ${partialShape}`);
  }
  if (notfullDefinedShape) {
    tensors.forEach(tensor => {
      partialShape = mergeElementShape(tensor.shape, partialShape);
    });
  }
  if (!fullDefinedShape(partialShape)) {
    throw new Error(`Non-fully-defined elementShape: ${partialShape}`);
  }
  return partialShape as number[];
}

export function mergeElementShape(
    elementShapeA: number|number[], elementShapeB: number|number[]): number|
    number[] {
  if (typeof elementShapeA === 'number') {
    return elementShapeB;
  }
  if (typeof elementShapeB === 'number') {
    return elementShapeA;
  }

  if (elementShapeA.length !== elementShapeB.length) {
    throw new Error(`Incompatible ranks during merge: ${elementShapeA} vs. ${
        elementShapeB}`);
  }

  const result: number[] = [];
  for (let i = 0; i < elementShapeA.length; ++i) {
    const dim0 = elementShapeA[i];
    const dim1 = elementShapeB[i];
    if (dim0 >= 0 && dim1 >= 0 && dim0 !== dim1) {
      throw new Error(`Incompatible shape during merge: ${elementShapeA} vs. ${
          elementShapeB}`);
    }
    result[i] = dim0 >= 0 ? dim0 : dim1;
  }
  return result;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {concat, DataType, keep, reshape, scalar, slice, stack, Tensor, tensor, tidy, unstack} from '@tensorflow/tfjs-core';

import {assertShapesMatchAllowUndefinedSize, inferElementShape, mergeElementShape} from './tensor_utils';

/**
 * TensorList stores a container of `tf.Tensor` objects, which are accessible
 * via tensors field.
 *
 * In order to get a copy of the underlying list, use the copy method:
 * ```
 *    TensorList b = a.copy();
 *    b.tensors().pushBack(t);  // This does not modify a.tensors().
 * ```
 *
 * Note that this is not a deep copy: the memory locations of the underlying
 * tensors will still point to the same locations of the corresponding tensors
 * in the original.
 */

export class TensorList {
  readonly idTensor: Tensor;
  maxNumElements: number;

  get id() {
    return this.idTensor.id;
  }
  /**
   *
   * @param tensors list of tensors
   * @param elementShape shape of each tensor, this can be a single number (any
   * shape is allowed) or partial shape (dim = -1).
   * @param elementDtype data type of each tensor
   * @param maxNumElements The maximum allowed size of `tensors`. Defaults to -1
   *   meaning that the size of `tensors` is unbounded.
   */
  constructor(
      readonly tensors: Tensor[], readonly elementShape: number|number[],
      readonly elementDtype: DataType, maxNumElements = -1) {
    if (tensors != null) {
      tensors.forEach(tensor => {
        if (elementDtype !== tensor.dtype) {
          throw new Error(`Invalid data types; op elements ${
              elementDtype}, but list elements ${tensor.dtype}`);
        }
        assertShapesMatchAllowUndefinedSize(
            elementShape, tensor.shape, 'TensorList shape mismatch: ');

        keep(tensor);
      });
    }
    this.idTensor = scalar(0);
    this.maxNumElements = maxNumElements;
    keep(this.idTensor);
  }

  /**
   * Get a new TensorList containing a copy of the underlying tensor container.
   */
  copy(): TensorList {
    return new TensorList(
        [...this.tensors], this.elementShape, this.elementDtype);
  }

  /**
   * Dispose the tensors and idTensor and clear the tensor list.
   */
  clearAndClose(keepIds?: Set<number>) {
    this.tensors.forEach(tensor => {
      if (keepIds == null || !keepIds.has(tensor.id)) {
        tensor.dispose();
      }
    });
    this.tensors.length = 0;
    this.idTensor.dispose();
  }
  /**
   * The size of the tensors in the tensor list.
   */
  size() {
    return this.tensors.length;
  }

  /**
   * Return a tensor that stacks a list of rank-R tf.Tensors into one rank-(R+1)
   * tf.Tensor.
   * @param elementShape shape of each tensor
   * @param elementDtype data type of each tensor
   * @param numElements the number of elements to stack
   */
  stack(elementShape: number[], elementDtype: DataType, numElements = -1):
      Tensor {
    if (elementDtype !== this.elementDtype) {
      throw new Error(`Invalid data types; op elements ${
          elementDtype}, but list elements ${this.elementDtype}`);
    }
    if (numElements !== -1 && this.tensors.length !== numElements) {
      throw new Error(`Operation expected a list with ${
          numElements} elements but got a list with ${
          this.tensors.length} elements.`);
    }
    assertShapesMatchAllowUndefinedSize(
        elementShape, this.elementShape, 'TensorList shape mismatch: ');
    const outputElementShape =
        inferElementShape(this.elementShape, this.tensors, elementShape);
    return tidy(() => {
      const reshapedTensors =
          this.tensors.map(tensor => reshape(tensor, outputElementShape));
      return stack(reshapedTensors, 0);
    });
  }

  /**
   * Pop a tensor from the end of the list.
   * @param elementShape shape of the tensor
   * @param elementDtype data type of the tensor
   */
  popBack(elementShape: number[], elementDtype: DataType): Tensor {
    if (elementDtype !== this.elementDtype) {
      throw new Error(`Invalid data types; op elements ${
          elementDtype}, but list elements ${this.elementDtype}`);
    }

    if (this.size() === 0) {
      throw new Error('Trying to pop from an empty list.');
    }
    const outputElementShape =
        inferElementShape(this.elementShape, this.tensors, elementShape);
    const tensor = this.tensors.pop();
    tensor.kept = false;

    assertShapesMatchAllowUndefinedSize(
        tensor.shape, elementShape, 'TensorList shape mismatch: ');

    return reshape(tensor, outputElementShape);
  }

  /**
   * Push a tensor to the end of the list.
   * @param tensor Tensor to be pushed.
   */
  pushBack(tensor: Tensor) {
    if (tensor.dtype !== this.elementDtype) {
      throw new Error(`Invalid data types; op elements ${
          tensor.dtype}, but list elements ${this.elementDtype}`);
    }

    assertShapesMatchAllowUndefinedSize(
        tensor.shape, this.elementShape, 'TensorList shape mismatch: ');

    if (this.maxNumElements === this.size()) {
      throw new Error(`Trying to push element into a full list.`);
    }
    keep(tensor);
    this.tensors.push(tensor);
  }

  /**
   * Update the size of the list.
   * @param size the new size of the list.
   */
  resize(size: number) {
    if (size < 0) {
      throw new Error(
          `TensorListResize expects size to be non-negative. Got: ${size}`);
    }

    if (this.maxNumElements !== -1 && size > this.maxNumElements) {
      throw new Error(`TensorListResize input size ${
          size} is greater maxNumElement ${this.maxNumElements}.`);
    }

    const destTensorList: TensorList = new TensorList(
        [], this.elementShape, this.elementDtype, this.maxNumElements);
    destTensorList.tensors.length = size;
    for (let i = 0; i < Math.min(this.tensors.length, size); ++i) {
      destTensorList.tensors[i] = this.tensors[i];
    }
    return destTensorList;
  }

  /**
   * Retrieve the element at the provided index
   * @param elementShape shape of the tensor
   * @param elementDtype dtype of the tensor
   * @param elementIndex index of the tensor
   */
  getItem(elementIndex: number, elementShape: number[], elementDtype: DataType):
      Tensor {
    if (elementDtype !== this.elementDtype) {
      throw new Error(`Invalid data types; op elements ${
          elementDtype}, but list elements ${this.elementDtype}`);
    }
    if (elementIndex < 0 || elementIndex > this.tensors.length) {
      throw new Error(`Trying to access element ${
          elementIndex} in a list with ${this.tensors.length} elements.`);
    }

    if (this.tensors[elementIndex] == null) {
      throw new Error(`element at index ${elementIndex} is null.`);
    }

    assertShapesMatchAllowUndefinedSize(
        this.tensors[elementIndex].shape, elementShape,
        'TensorList shape mismatch: ');
    const outputElementShape =
        inferElementShape(this.elementShape, this.tensors, elementShape);
    return reshape(this.tensors[elementIndex], outputElementShape);
  }

  /**
   * Set the tensor at the index
   * @param elementIndex index of the tensor
   * @param tensor the tensor to be inserted into the list
   */
  setItem(elementIndex: number, tensor: Tensor) {
    if (tensor.dtype !== this.elementDtype) {
      throw new Error(`Invalid data types; op elements ${
          tensor.dtype}, but list elements ${this.elementDtype}`);
    }

    if (elementIndex < 0 ||
        this.maxNumElements !== -1 && elementIndex >= this.maxNumElements) {
      throw new Error(`Trying to set element ${
          elementIndex} in a list with max ${this.maxNumElements} elements.`);
    }

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, tensor.shape, 'TensorList shape mismatch: ');
    keep(tensor);

    // dispose the previous value if it is replacing.
    if (this.tensors[elementIndex] != null) {
      this.tensors[elementIndex].kept = false;
    }

    this.tensors[elementIndex] = tensor;
  }

  /**
   * Return selected values in the TensorList as a stacked Tensor. All of
   * selected values must have been written and their shapes must all match.
   * @param indices indices of tensors to gather
   * @param elementDtype output tensor dtype
   * @param elementShape output tensor element shape
   */
  gather(indices: number[], elementDtype: DataType, elementShape: number[]):
      Tensor {
    if (elementDtype !== this.elementDtype) {
      throw new Error(`Invalid data types; op elements ${
          elementDtype}, but list elements ${this.elementDtype}`);
    }

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, elementShape, 'TensorList shape mismatch: ');

    // When indices is greater than the size of the list, indices beyond the
    // size of the list are ignored.
    indices = indices.slice(0, this.size());
    const outputElementShape =
        inferElementShape(this.elementShape, this.tensors, elementShape);
    if (indices.length === 0) {
      return tensor([], [0].concat(outputElementShape));
    }

    return tidy(() => {
      const tensors =
          indices.map(i => reshape(this.tensors[i], outputElementShape));
      return stack(tensors, 0);
    });
  }

  /**
   * Return the values in the TensorList as a concatenated Tensor.
   * @param elementDtype output tensor dtype
   * @param elementShape output tensor element shape
   */
  concat(elementDtype: DataType, elementShape: number[]): Tensor {
    if (!!elementDtype && elementDtype !== this.elementDtype) {
      throw new Error(`TensorList dtype is ${
          this.elementDtype} but concat requested dtype ${elementDtype}`);
    }

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, elementShape, 'TensorList shape mismatch: ');
    const outputElementShape =
        inferElementShape(this.elementShape, this.tensors, elementShape);

    if (this.size() === 0) {
      return tensor([], [0].concat(outputElementShape));
    }
    return tidy(() => {
      const tensors = this.tensors.map(t => reshape(t, outputElementShape));
      return concat(tensors, 0);
    });
  }
}

/**
 * Creates a TensorList which, when stacked, has the value of tensor.
 * @param tensor from tensor
 * @param elementShape output tensor element shape
 */
export function fromTensor(
    tensor: Tensor, elementShape: number[], elementDtype: DataType) {
  const dtype = tensor.dtype;
  if (tensor.shape.length < 1) {
    throw new Error(
        `Tensor must be at least a vector, but saw shape: ${tensor.shape}`);
  }
  if (tensor.dtype !== elementDtype) {
    throw new Error(`Invalid data types; op elements ${
        tensor.dtype}, but list elements ${elementDtype}`);
  }
  const tensorElementShape = tensor.shape.slice(1);
  assertShapesMatchAllowUndefinedSize(
      tensorElementShape, elementShape, 'TensorList shape mismatch: ');
  const tensorList: Tensor[] = unstack(tensor);
  return new TensorList(tensorList, elementShape, dtype);
}

/**
 * Return a TensorList of the given size with empty elements.
 * @param elementShape the shape of the future elements of the list
 * @param elementDtype the desired type of elements in the list
 * @param numElements the number of elements to reserve
 * @param maxNumElements the maximum number of elements in th list
 */
export function reserve(
    elementShape: number[], elementDtype: DataType, numElements: number,
    maxNumElements: number) {
  return new TensorList([], elementShape, elementDtype, maxNumElements);
}

/**
 * Put tensors at specific indices of a stacked tensor into a TensorList.
 * @param indices list of indices on how to scatter the tensor.
 * @param tensor input tensor.
 * @param elementShape the shape of the future elements of the list
 * @param numElements the number of elements to scatter
 */
export function scatter(
    tensor: Tensor, indices: number[], elementShape: number[],
    numElements?: number): TensorList {
  if (indices.length !== tensor.shape[0]) {
    throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${
        indices.length} vs. ${tensor.shape[0]}`);
  }

  const maxIndex = Math.max(...indices);

  if (numElements != null && numElements !== -1 && maxIndex >= numElements) {
    throw new Error(
        `Max index must be < array size (${maxIndex}  vs. ${numElements})`);
  }

  const list = new TensorList([], elementShape, tensor.dtype, numElements);
  const tensors = unstack(tensor, 0);
  indices.forEach((value, index) => {
    list.setItem(value, tensors[index]);
  });
  return list;
}

/**
 * Split the values of a Tensor into a TensorList.
 * @param length the lengths to use when splitting value along
 *    its first dimension.
 * @param tensor the tensor to split.
 * @param elementShape the shape of the future elements of the list
 */
export function split(
    tensor: Tensor, length: number[], elementShape: number[]) {
  let totalLength = 0;
  const cumulativeLengths = length.map(len => {
    totalLength += len;
    return totalLength;
  });

  if (totalLength !== tensor.shape[0]) {
    throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${totalLength}, and tensor's shape is: ${tensor.shape}`);
  }

  const shapeWithoutFirstDim = tensor.shape.slice(1);
  const outputElementShape =
      mergeElementShape(shapeWithoutFirstDim, elementShape);
  const elementPerRow = totalLength === 0 ? 0 : tensor.size / totalLength;
  const tensors: Tensor[] = tidy(() => {
    const tensors = [];
    tensor = reshape(tensor, [1, totalLength, elementPerRow]);
    for (let i = 0; i < length.length; ++i) {
      const previousLength = (i === 0) ? 0 : cumulativeLengths[i - 1];
      const indices = [0, previousLength, 0];
      const sizes = [1, length[i], elementPerRow];
      tensors[i] = reshape(
          slice(tensor, indices, sizes), outputElementShape as number[]);
    }
    tensor.dispose();
    return tensors;
  });

  const list = new TensorList([], elementShape, tensor.dtype, length.length);

  for (let i = 0; i < tensors.length; i++) {
    list.setItem(i, tensors[i]);
  }
  return list;
}

