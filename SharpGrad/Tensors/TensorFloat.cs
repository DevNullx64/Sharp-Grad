using ILGPU.Runtime;
using SharpGrad.Memory;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace SharpGrad.Tensors
{
    interface ITensor<TSelf, T>:
        IAdditionOperators<TSelf, Tensor<T>, Tensor<T>>,
        ISubtractionOperators<TSelf, Tensor<T>, Tensor<T>>,
        IUnaryNegationOperators<TSelf, Tensor<T>>,
        IMultiplyOperators<TSelf, Tensor<T>, Tensor<T>>,
        IDivisionOperators<TSelf, Tensor<T>, Tensor<T>>
        where T: unmanaged, INumber<T>
        where TSelf : ITensor<TSelf, T>
    { }

    internal class Tensor<T> : ITensor<Tensor<T>, T>
        where T : unmanaged, INumber<T>
    {
    }
}
