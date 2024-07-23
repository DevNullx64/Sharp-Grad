using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public class KpuExecScript<T> : IReadOnlyList<OperationKPU>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        private readonly OperationKPU[] operations = [];
        public OperationKPU this[int index] => operations[index];

        private readonly TensorData<T>[] datas = [];
        public IReadOnlyList<TensorData<T>> Datas => datas;
        public readonly byte CacheSize;

        public int Count => operations.Length;

        public KpuExecScript(IList<OperationKPU> operations, IList<ITensor<T>> datas, byte registersCount)
        {
            this.operations = [.. operations];
            this.datas = datas.Cast<TensorData<T>>().ToArray();
            CacheSize = registersCount;
        }

        public IEnumerator<OperationKPU> GetEnumerator() => ((IEnumerable<OperationKPU>)operations).GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => operations.GetEnumerator();
    }
}