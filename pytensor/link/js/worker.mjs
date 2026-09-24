import fs from 'node:fs';
import vm from 'node:vm';

const input=0;
const output=1;

function read(buffer){
  let n=0;
  while(n<buffer.length){
    const got=fs.readSync(input,buffer,n,buffer.length-n,null);
    if(got===0)throw Error('Python closed the evaluator pipe');
    n+=got;
  }
}
function write(buffer){
  let n=0;
  while(n<buffer.length)n+=fs.writeSync(output,buffer,n,buffer.length-n);
}
function strides(shape){
  let p=1;const result=Array(shape.length);
  for(let k=shape.length-1;k>=0;k--){result[k]=p;p*=shape[k]}
  return result;
}
function packed(rec){
  const expected=strides(rec.s);
  if(expected.every((v,k)=>v===rec.t[k]))return Buffer.from(rec.d.buffer,rec.d.byteOffset,rec.d.byteLength);
  const count=rec.s.reduce((a,b)=>a*b,1);
  const result=Buffer.allocUnsafe(count*8);
  for(let flat=0;flat<count;flat++){
    let rest=flat,address=0;
    for(let axis=0;axis<rec.s.length;axis++){
      const coordinate=Math.floor(rest/expected[axis]);
      rest-=coordinate*expected[axis];
      address+=coordinate*rec.t[axis];
    }
    result.writeDoubleLE(rec.d[address],flat*8);
  }
  return result;
}
function respond(status,parts){
  const length=parts.reduce((n,part)=>n+part.length,0);
  const header=Buffer.allocUnsafe(5);
  header.writeUInt8(status,0);header.writeUInt32LE(length,1);
  write(Buffer.concat([header,...parts],5+length));
}
try {
  const length=Buffer.allocUnsafe(4);
  read(length);
  const init=Buffer.allocUnsafe(length.readUInt32LE());
  read(init);
  const spec=JSON.parse(init.toString('utf8'));
  vm.runInThisContext(spec.source);
  for(const [k,item] of spec.constants.entries()){
    const raw=Buffer.from(item.data,'base64');
    const data=new Float64Array(raw.length/8);
    Buffer.from(data.buffer).set(raw);
    __ptjs.constants[k]={d:data,s:item.shape,t:strides(item.shape)};
  }
  write(Buffer.from([1]));
  const rankIn=spec.inputs;
  const rankOut=spec.outputs;
  while(true){
    const header=Buffer.allocUnsafe(5);
    read(header);
    const command=header.readUInt8(0);
    const payload=Buffer.allocUnsafe(header.readUInt32LE(1));
    read(payload);
    if(command===0)break;
    if(command===2){
      const n=payload.readUInt32LE();
      const start=process.hrtime.bigint();
      __ptjs.repeat(n);
      const elapsed=Number(process.hrtime.bigint()-start)/n/1000;
      const response=Buffer.allocUnsafe(8);
      response.writeDoubleLE(elapsed);
      respond(1,[response]);
      continue;
    }
    if(command!==1)throw Error('unknown evaluator command');
    let offset=0;
    for(let k=0;k<rankIn.length;k++){
      const rank=rankIn[k];
      const shape=Array.from({length:rank},(_,a)=>payload.readInt32LE(offset+a*4));
      offset+=rank*4;
      if(shape.some(d=>d<0))throw Error('negative shape');
      __ptjs.setInputShape(k,shape);
      const rec=__ptjs.inputs[k];
      const bytes=rec.d.byteLength;
      payload.copy(Buffer.from(rec.d.buffer),0,offset,offset+bytes);
      offset+=bytes;
    }
    if(offset!==payload.length)throw Error('input frame length mismatch');
    try{
      __ptjs.execute();
      const result=rankOut.flatMap((rank,k)=>{
        const rec=__ptjs.outputs()[k];
        if(rec.s.length!==rank)throw Error('output rank changed');
        const dims=Buffer.allocUnsafe(rank*4);
        for(let a=0;a<rank;a++)dims.writeInt32LE(rec.s[a],a*4);
        return [dims,packed(rec)];
      });
      respond(1,result);
    }catch(err){
      const message=Buffer.from(String(err.stack||err));
      respond(0,[message]);
    }
  }
}catch(err){
  if(err.message!=='Python closed the evaluator pipe'){
    process.stderr.write(String(err.stack||err)+'\n');
    process.exitCode=1;
  }
}
