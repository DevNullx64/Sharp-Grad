using ILGPU;
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
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        Shape Shape { get; set; }
        long Length { get; }
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
        public long Length => Shape.Length;

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
        protected readonly AcceleratorBuffer<T> buffer;
        internal ArrayView1D<T, Stride1D.Dense> view => buffer.AcceleratorData.View;
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
            buffer = KernelProcessUnit.DefaultKPU.GetBuffer<T>(shape.Length);
        }

        public DataTensor(Shape shape, T[] data) : base(shape)
        {
            if (shape.Length != data.Length)
                throw new InvalidOperationException($"Invalid data length {data.Length} for shape {shape}");
            Shape = shape;
            buffer = KernelProcessUnit.DefaultKPU.GetBuffer(data);
        }
    }

    public interface ITensorOperation<T> : ITensor<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        OpCode OpCode { get; }

        void DepthFirstSearch(List<ITensorOperation<T>> topoSort, Dictionary<Tensor<T>, int> visited, Dictionary<Tensor<T>, int> leaf);
        void Backward();
    }
    internal interface ITensorOperation1<T> : ITensorOperation<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        Tensor<T> Operand1 { get; }
    }
    internal interface ITensorOperation1<T, TOp> : ITensorOperation1<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IExecutor1<T, T>
    { }

    internal interface ITensorOperation2<T> : ITensorOperation<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        Tensor<T> Operand1 { get; }
        Tensor<T> Operand2 { get; }
    }
    internal interface ITensorOperation2<T, TOp> : ITensorOperation2<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IExecutor2<T, T, T>
    { }

    internal interface ITensorReduce<T> : ITensorOperation1<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    { }

    internal interface ITensorReduce<T, TOp> : ITensorReduce<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IAggregator<T, TOp>
    { }


    internal abstract class Tensor<T, TOp>(Shape shape) : Tensor<T>(shape), ITensorOperation<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IExecutor
    {
        public OpCode OpCode => TOp.OpCode;

        public abstract void DepthFirstSearch(List<ITensorOperation<T>> topoSort, Dictionary<Tensor<T>, int> visited, Dictionary<Tensor<T>, int> leaf);
        public abstract void Backward();
    }

    internal class StreamTensor1<T, TOp>(Tensor<T> operand1)
        : Tensor<T, TOp>(operand1.Shape), ITensorOperation1<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IExecutor1<T, T>
    {
        public Tensor<T> Operand1 => operand1;

        public override T this[params Index[] indices] => TOp.Exec(operand1[indices]);

        public override void DepthFirstSearch(List<ITensorOperation<T>> topoSort, Dictionary<Tensor<T>, int> visited, Dictionary<Tensor<T>, int> leaf)
        {
            if (visited.TryGetValue(this, out var count1))
            {
                visited[this] = count1 + 1;
            }
            else
            {
                visited.Add(this, 1);
                if (Operand1 is ITensorOperation<T> op)
                    op.DepthFirstSearch(topoSort, visited, leaf);
                else
                {
                    if (leaf.TryGetValue(Operand1, out var count2))
                        leaf[Operand1] = count2 + 1;
                    else
                        leaf.Add(Operand1, 1);
                }
                topoSort.Add(this);
            }
        }

        public override void Backward()
        {
            if (Operand1 is ITensorOperation<T> op)
                op.Backward();
        }
    }

    internal class StreamTensor2<T, TOp>(Tensor<T> operand1, Tensor<T> operand2)
        : Tensor<T, TOp>(operand1.Shape), ITensorOperation2<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IExecutor2<T, T, T>
    {
        public Tensor<T> Operand1 => operand1;
        public Tensor<T> Operand2 => operand2;

        public override T this[params Index[] indices] => TOp.Exec(operand1[indices], operand2[indices]);

        public override void DepthFirstSearch(List<ITensorOperation<T>> topoSort, Dictionary<Tensor<T>, int> visited, Dictionary<Tensor<T>, int> leaf)
        {
            if (visited.TryGetValue(this, out var count1))
            {
                visited[this] = count1 + 1;
            }
            else
            {
                visited.Add(this, 1);
                if (Operand1 is ITensorOperation<T> op1)
                    op1.DepthFirstSearch(topoSort, visited, leaf);
                else
                {
                    if (leaf.TryGetValue(Operand1, out var count2))
                        leaf[Operand1] = count2 + 1;
                    else
                        leaf.Add(Operand1, 1);
                }
                if (Operand2 is ITensorOperation<T> op2)
                    op2.DepthFirstSearch(topoSort, visited, leaf);
                else
                {
                    if (leaf.TryGetValue(Operand2, out var count3))
                        leaf[Operand2] = count3 + 1;
                    else
                        leaf.Add(Operand2, 1);
                }
                topoSort.Add(this);
            }
        }

        public override void Backward()
        {
            if (Operand1 is ITensorOperation<T> op1)
                op1.Backward();
            if (Operand2 is ITensorOperation<T> op2)
                op2.Backward();
        }
    }

    internal class StreamAggregator<T, TOp>(Tensor<T> operand1)
        : Tensor<T, TOp>(operand1.Shape), ITensorReduce<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IAggregator<T, TOp>
    {
        public Tensor<T> Operand1 => throw new NotImplementedException();
        public override T this[params Index[] indices] => throw new NotImplementedException();

        public override void DepthFirstSearch(List<ITensorOperation<T>> topoSort, Dictionary<Tensor<T>, int> visited, Dictionary<Tensor<T>, int> leaf)
        {
            if (visited.TryGetValue(this, out var count1))
            {
                visited[this] = count1 + 1;
            }
            else
            {
                visited.Add(this, 1);
                if (Operand1 is ITensorOperation<T> op1)
                    op1.DepthFirstSearch(topoSort, visited, leaf);
                else
                {
                    if (leaf.TryGetValue(Operand1, out var count2))
                        leaf[Operand1] = count2 + 1;
                    else
                        leaf.Add(Operand1, 1);
                }
                topoSort.Add(this);
            }
        }

        public override void Backward()
        {
            if (Operand1 is ITensorOperation<T> op1)
                op1.Backward();
        }
    }
}
