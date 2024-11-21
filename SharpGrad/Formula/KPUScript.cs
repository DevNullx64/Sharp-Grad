using SharpGrad.Memory;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Formula
{
    public readonly struct KPUScript<TResult>
        where TResult : unmanaged, INumber<TResult>
    {
        internal readonly AcceleratorBuffer<TResult>[] datas;
        internal readonly AcceleratorBuffer<TResult>[] gradients;
        internal readonly AcceleratorBuffer<TResult>[] Outputs;


        internal readonly DataInfo<TResult>[] dataInfos;
        public readonly IReadOnlyList<DataInfo<TResult>> Gradients => dataInfos;

        internal readonly OperationInfo<TResult>[] operations;
        public readonly IReadOnlyList<OperationInfo<TResult>> Operations => operations;


        internal KPUScript(ScriptBuilder<TResult> script)
        {
            datas = [.. script.Datas];
            gradients = [.. script.Gradients];
            dataInfos = [.. script.DataInfos];
            operations = [.. script.Operations];
            Outputs = [.. script.Outputs];
        }

        public static implicit operator KPUScript<TResult>(ScriptBuilder<TResult> script)
            => new(script);
    }
}