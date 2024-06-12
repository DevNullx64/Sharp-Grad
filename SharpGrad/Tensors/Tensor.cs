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
    public interface ITensor<T>
        where T: unmanaged, INumber<T>, IPowerFunctions<T>
    {
        Shape Shape { get; set; }
        T this[params Index[] indices] { get; }
    }

    public abstract class Tensor<T>(Shape shape) : ITensor<T>,
        IAdditionOperators<Tensor<T>, Tensor<T>, Tensor<T>>,
        ISubtractionOperators<Tensor<T>, Tensor<T>, Tensor<T>>,
        IUnaryNegationOperators<Tensor<T>, Tensor<T>>,
        IMultiplyOperators<Tensor<T>, Tensor<T>, Tensor<T>>,
        IDivisionOperators<Tensor<T>, Tensor<T>, Tensor<T>>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
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

        public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right) => new StreamTensor2<T, AddOp<T>>(left, right);
        public static Tensor<T> operator -(Tensor<T> value) => new StreamTensor1<T, NegOp<T>>(value);
        public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right) => new StreamTensor2<T, SubOp<T>>(left, right);
        public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right) => new StreamTensor2<T, MulOp<T>>(left, right);
        public static Tensor<T> operator /(Tensor<T> left, Tensor<T> right) => new StreamTensor2<T, DivOp<T>>(left, right);


        public static implicit operator Tensor<T>(T[] data) => new DataTensor<T>(new Shape(data.Length), data);
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

        public DataTensor(Shape shape) : base(shape)
        {
            Shape = shape;
            buffer = DefaultKpu.GetAcceleratorBuffer<T>(shape.Length);
        }

        public DataTensor(Shape shape, T[] data) : base(shape)
        {
            if (shape.Length != data.Length)
                throw new InvalidOperationException($"Invalid data length {data.Length} for shape {shape}");
            Shape = shape;
            buffer = KernelProcessUnit.GetAcceleratorBuffer(data);
        }
    }

    internal interface ITensorOperation<T, TOp>
         where T : unmanaged, INumber<T>
         where TOp : IExecutor
    {
        OpCode OpCode { get; }
    }
    internal interface ITensorOperation1<T, TOp>
        : ITensorOperation<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IExecutor1<T, T>
    {
        Tensor<T> Operand1 { get; }
    }
    internal interface ITensorOperation2<T, TOp> : ITensorOperation<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IExecutor2<T, T, T>
    {
        Tensor<T> Operand1 { get; }
        Tensor<T> Operand2 { get; }
    }
    internal interface ITensorAggregator<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IAggregator<T, TOp>
    {
        Tensor<T> Operand1 { get; }
    }


    internal abstract class Tensor<T, TOp>(Shape shape) : Tensor<T>(shape), ITensorOperation<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IExecutor
    {
        public OpCode OpCode => TOp.OpCode;
    }

    internal class StreamTensor1<T, TOp>(Tensor<T> operand1)
        : Tensor<T, TOp>(operand1.Shape), ITensorOperation1<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IExecutor1<T, T>
    {
        public Tensor<T> Operand1 => operand1;

        public override T this[params Index[] indices] => TOp.Exec(operand1[indices]);
    }

    internal class StreamTensor2<T, TOp>(Tensor<T> operand1, Tensor<T> operand2)
        : Tensor<T, TOp>(operand1.Shape), ITensorOperation2<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IExecutor2<T, T, T>
    {
        public Tensor<T> Operand1 => operand1;
        public Tensor<T> Operand2 => operand2;

        public override T this[params Index[] indices] => TOp.Exec(operand1[indices], operand2[indices]);
    }

    internal class StreamAggregator<T, TOp>(Tensor<T> operand1)
        : Tensor<T, TOp>(operand1.Shape), ITensorAggregator<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IAggregator<T, TOp>
    {
        public Tensor<T> Operand1 => throw new NotImplementedException();
        public override T this[params Index[] indices] => throw new NotImplementedException();
    }
}
