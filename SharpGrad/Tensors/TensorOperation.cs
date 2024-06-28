using SharpGrad.Tensors.Operators;
using System.Linq;
using System.Numerics;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace SharpGrad.Tensors
{
    internal abstract class TensorOperation<T, TOp>(Shape shape) : Tensor<T>(TOp.Symbol, shape), ITensorOperation<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        where TOp : IExecutor
    {
        public OpCode OpCode => TOp.OpCode;

        public abstract int OperandCound { get; }
    }
}
