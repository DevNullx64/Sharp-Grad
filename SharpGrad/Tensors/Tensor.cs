using ILGPU.Runtime;
using SharpGrad.Memory;
using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace SharpGrad.Tensors
{
    public interface ITensor<TSelf, T>:
        IAdditionOperators<TSelf, Tensor<T>, Tensor<T>>,
        ISubtractionOperators<TSelf, Tensor<T>, Tensor<T>>,
        IUnaryNegationOperators<TSelf, Tensor<T>>,
        IMultiplyOperators<TSelf, Tensor<T>, Tensor<T>>,
        IDivisionOperators<TSelf, Tensor<T>, Tensor<T>>
        where T: unmanaged, INumber<T>
        where TSelf : ITensor<TSelf, T>
    {
        Shape Shape { get; set; }
        T this[params Index[] indices] { get; }
    }

    public abstract class Tensor<T>(Shape shape) : ITensor<Tensor<T>, T>
        where T : unmanaged, INumber<T>
    {
        protected Shape Shape_ = shape;
        public virtual Shape Shape
        {
            get => Shape_;
            set
            {
                if (Shape_.Length != value.Length)
                    throw new InvalidOperationException($"Cannot update shapes {Shape_} and {value}");
                Shape_ = value;
            }
        }

        public abstract T this[params Index[] indices] { get; }

        public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right)
        {
            throw new NotImplementedException();
        }

        public static Tensor<T> operator -(Tensor<T> value)
        {
            throw new NotImplementedException();
        }

        public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right)
        {
            throw new NotImplementedException();
        }

        public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right)
        {
            throw new NotImplementedException();
        }

        public static Tensor<T> operator /(Tensor<T> left, Tensor<T> right)
        {
            throw new NotImplementedException();
        }
    }

    internal class DataTensor<T> : Tensor<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        private readonly AcceleratorBuffer<T> buffer;
        public override T this[params Index[] indices]
        {
            get
            {
                var flattenedIndex = Shape.FlattenFrom(Shape, indices);
                return buffer[flattenedIndex];
            }
        }

        public DataTensor(Shape shape): base(shape)
        {
            buffer = Acc.GetAcceleratorBuffer<T>(shape.Length);
            Shape = shape;
        }

        public DataTensor(Shape shape, T[] data) : base(shape)
        {
            buffer = Acc.GetAcceleratorBuffer(data);
            Shape = shape;
        }
    }


    internal abstract class TensorOperation<T>(OpCode opCode)
        where T : unmanaged, INumber<T>
    {
        public readonly OpCode OpCode = opCode;
    }
    internal class TensorOperation1<T>(OpCode opCode, Tensor<T> operand1) : TensorOperation<T>(opCode)
        where T : unmanaged, INumber<T>
    {
        public readonly Tensor<T> Operand1 = operand1;
    }
    internal class TensorOperation2<T>(OpCode opCode, Tensor<T> operand1, Tensor<T> operand2) : TensorOperation<T>(opCode)
        where T : unmanaged, INumber<T>
    {
        public readonly Tensor<T> Operand1 = operand1;
        public readonly Tensor<T> Operand2 = operand2;
    }


    internal abstract class ComputedTensor<T>(Shape shape) : Tensor<T>(shape)
        where T : unmanaged, INumber<T>
    {
        internal abstract IEnumerable<TensorOperation<T>> operations { get; }
    }

    internal abstract class ComputedTensor1<T>(OpCode opCode, Tensor<T> operand1) : ComputedTensor<T>(operand1.Shape)
        where T : unmanaged, INumber<T>
    {
        public readonly TensorOperation1<T> TensorOperation = opCode.HasFlag(OpCode.Unary)
            ? new(opCode, operand1)
            : throw new InvalidOperationException($"Invalid unary operation {opCode}");

        internal override IEnumerable<TensorOperation<T>> operations 
            => operand1 is ComputedTensor<T> computed
            ? computed.operations.Append(TensorOperation)
            : (IEnumerable<TensorOperation<T>>)([TensorOperation]);

    }
}
