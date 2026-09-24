import fs from 'node:fs';
import vm from 'node:vm';

const [sourcePath, inputsPath] = process.argv.slice(2);
vm.runInThisContext(fs.readFileSync(sourcePath, 'utf8'));
const spec = JSON.parse(fs.readFileSync(inputsPath, 'utf8'));
for (const [i, input] of spec.constants.entries()) {
  const shape = input.shape;
  let size = 1;
  const stride = Array(shape.length);
  for (let j=shape.length-1;j>=0;j--) {stride[j]=size;size*=shape[j]}
  __ptjs.constants[i]={d:new Float64Array(input.data),s:shape,t:stride};
}
const results=[];
for (const call of spec.calls) {
  for (const [i, input] of call.entries()) {
    __ptjs.setInputShape(i, input.shape);
    __ptjs.inputs[i].d.set(input.data);
  }
  const shapes=JSON.parse(__ptjs.run());
  results.push(shapes.map((shape,i)=>({shape,data:Array.from(__ptjs.outputs()[i].d)})));
}
console.log(JSON.stringify(results));
