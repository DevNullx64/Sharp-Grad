using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Operators;
using System.Numerics;
using System.Threading.Tasks;

namespace SharpGrad.Formula
{
    public static class AcceleratorExecutor
    {
        public static readonly Context context = Context.Create(builder => builder.AllAccelerators());
        public static readonly Device device = context.GetPreferredDevice(preferCPU: false);
        public static readonly Accelerator Accelerator = device.CreateAccelerator(context);

        private static T OperationInvoke<T>(OpCode opCode, T left)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
            => opCode switch
            {
                OpCode.Neg => NegOp<T>.Invoke(left),
                OpCode.Log => LogOp<T>.Invoke(left),
                OpCode.Exp => ExpOp<T>.Invoke(left),
                //OpCode.Sqrt => SqrtOp<T>.Invoke(left),
                //OpCode.Sin => SinOp<T>.Invoke(left),
                //OpCode.Cos => CosOp<T>.Invoke(left),
                //OpCode.Tan => TanOp<T>.Invoke(left),
                _ => default // TODO: This should not happen
            };

        private static T OperationInvoke<T>(OpCode opCode, T left, T right)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
            => opCode switch
            {
                OpCode.Add => AddOp<T>.Invoke(left, right),
                OpCode.Sub => SubOp<T>.Invoke(left, right),
                OpCode.Mul => MulOp<T>.Invoke(left, right),
                OpCode.Div => DivOp<T>.Invoke(left, right),
                OpCode.Pow => PowOp<T>.Invoke(left, right),
                _ => default // TODO: This should not happen
            };

        private static T Forward<T>(
            Index1D idx,
            OperationInfo<T> op,
            ArrayView2D<T, Stride2D.DenseY> datas,
            T[] cache)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            T left = op.LeftIndex.Category == SourceOfOperand.Data
                ? datas[op.LeftIndex.Index, idx]
                : cache[op.LeftIndex.Index];

            if (op.RightIndex.IsEmpty)
            {
                return OperationInvoke(op.OpCode, left);
            }
            else
            {
                T right = op.RightIndex.Category == SourceOfOperand.Data
                    ? datas[op.RightIndex.Index, idx]
                    : cache[op.RightIndex.Index];
                return OperationInvoke(op.OpCode, left, right);
            }
        }

        private static void ForwardKernel<T>(
            Index1D idx,
            ArrayView<OperationInfo<T>> ops,
            ArrayView2D<T, Stride2D.DenseY> datas,
            ArrayView2D<T, Stride2D.DenseY> outputs,
            SpecializedValue<ushort> cacheSize)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            T[] cache = new T[cacheSize];
            for (int i = 0; i < cacheSize; i++)
            {
                var op = ops[i];
                cache[op.OutputIndex] = Forward(idx, op, datas, cache);
                if (!op.OutputIndex.IsEmpty)
                    outputs[(int)op.OutputIndex, idx] = cache[op.OutputIndex];
            }
        }

        /*
        internal static void Forward<T>(
            SafeAccelerator accelerator,
            OperationInfo<T>[] ops,
            AcceleratorBuffer<T>[] datas,
            out AcceleratorBuffer<T>[] outputs)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            if(datas.Any(d => d.SafeAccelerator != accelerator))
                throw new ArgumentException("All data buffers must be on the same accelerator.");


        }
        */

        private static T Backward<T>(OpCode opCode, T left, T currentGrad)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
            => opCode switch
            {
                OpCode.Neg => NegOp<T>.Backward(left, currentGrad),
                OpCode.Log => LogOp<T>.Backward(left, currentGrad),
                OpCode.Exp => ExpOp<T>.Backward(left, currentGrad),
                _ => default // TODO: This should not happen
            };

        private static (T leftGrad, T rightGrad) Backward<T>(OpCode opCode, T left, T right, T currentGrad)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
            => opCode switch
            {
                OpCode.Add => AddOp<T>.Backward(left, right, currentGrad),
                OpCode.Sub => SubOp<T>.Backward(left, right, currentGrad),
                OpCode.Mul => MulOp<T>.Backward(left, right, currentGrad),
                OpCode.Div => DivOp<T>.Backward(left, right, currentGrad),
                OpCode.Pow => PowOp<T>.Backward(left, right, currentGrad),
                _ => default // TODO: This should not happen
            };

        private static void BackwardKernel<T>(
            Index1D idx,
            ArrayView<OperationInfo<T>> ops,
            ArrayView2D<T, Stride2D.DenseY> datas,
            ArrayView1D<BIndex<ushort>, Stride1D.Dense> dataGradIndices,
            ArrayView1D<T, Stride1D.Dense> outputs,
            ArrayView1D<T, Stride1D.Dense> grads,
            SpecializedValue<ushort> cacheSize)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>, IPowerFunctions<T>
        {
            T[] cache = new T[cacheSize];
            T[] gradCache = new T[cacheSize];

            // Forward pass
            for (int i = 0; i < ops.Length; i++)
            {
                var op = ops[i];
                cache[op.OutputIndex] = Forward(idx, op, datas, cache);
                if (!op.OutputIndex.IsEmpty)
                    outputs[op.OutputIndex] = cache[op.OutputIndex];
                gradCache[i] = !op.GradientIndex.IsEmpty
                    ? grads[op.GradientIndex]
                    : T.Zero;
            }

            // Backward pass
            for (int i = cacheSize - 1; i >= 0; i--)
            {
                var op = ops[i];
                T currentGrad = gradCache[i];

                T left = op.LeftIndex.Category == SourceOfOperand.Data
                    ? datas[idx, op.LeftIndex.Index]
                    : cache[op.LeftIndex.Index];

                T leftGrad;
                if (op.RightIndex.IsEmpty)
                {
                    leftGrad = Backward(op.OpCode, left, currentGrad);
                }
                else
                {
                    T right = op.RightIndex.Category == SourceOfOperand.Data
                        ? datas[idx, op.RightIndex.Index]
                        : cache[op.RightIndex.Index];

                    (leftGrad, T rightGrad) = Backward(op.OpCode, left, right, currentGrad);

                    if (op.RightIndex.Category == SourceOfOperand.Data)
                        grads[op.RightIndex.Index] += rightGrad;
                    else
                        gradCache[op.RightIndex.Index] += rightGrad;
                }

                if (op.LeftIndex.Category == SourceOfOperand.Data)
                    grads[op.LeftIndex.Index] += leftGrad;
                else
                    gradCache[op.LeftIndex.Index] += leftGrad;
            }
        }

        public static async Task ForwardAsync<T>(AcceleratorExtender accelerator, KPUScript<T> script)
        where T : unmanaged, INumber<T>
        {
            foreach (var data in script.datas)
            {
                var b = data.IsExclusiveLockHeld; // TODO: This should be async
                data.SafeAccelerator = accelerator;
            }
        }
    }
}
