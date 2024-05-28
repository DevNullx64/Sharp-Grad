using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackwardable { }
    public interface IBackwardableOne<TType, TGrad, TOp>: IBackwardable
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
        where TOp : IBackwardOne<TType, TGrad>
    {
        public void BackwardCpu();
    }

    public interface IBackwardableTwo<TType, TGrad, TOp> : IBackwardable
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
        where TOp : IBackwardTwo<TType, TGrad>
    {
        public void Backward();
    }

    internal class TensorOneOp<TType, TGrad, TOp>(Shape shape, TensorBase<TType, TGrad> left) : TensorBase<TType, TGrad>(shape), IBackwardableOne<TType, TGrad, TOp>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
        where TOp : IBackwardOne<TType, TGrad>
    {
        public readonly TensorBase<TType, TGrad> Left = left;
        private readonly Tensor<TType, TGrad>? LeftT = left as Tensor<TType, TGrad>;

        private readonly AcceleratorBuffer<TType> data = new((int)shape.Size);
        internal override AcceleratorBuffer<TType> Data => data;
        public AcceleratorBuffer<TGrad> Grad => new((int)shape.Size);

        public override TType this[params int[] indices] {
            get => TOp.ApplyCpu(Left[indices]);
            set => throw new System.NotImplementedException($"Cannot set value to {GetType().Name}");
        }

        public void BackwardCpu()
        {
            if (LeftT is null)
                return;

            for(int i = 0; i < Length; i++)
            {

                //LeftT.AddGradient(TOp.BackwardCpu(Grad.CPUData[i], LeftT.Data.CPUData[i]));

            }
        }
    }

    internal class TensorTwoOp<TType, TGrad, TOp>(Shape shape, TensorBase<TType, TGrad> left, TensorBase<TType, TGrad> right) : TensorBase<TType, TGrad>(shape)
    where TType : unmanaged, INumber<TType>
    where TGrad : unmanaged, IFloatingPoint<TGrad>
    where TOp : IBackwardTwo<TType, TGrad>
    {
        public readonly TensorBase<TType, TGrad> Left = left;
        public readonly TensorBase<TType, TGrad> Right = right;

        private readonly AcceleratorBuffer<TType> data = new((int)shape.Size);
        internal override AcceleratorBuffer<TType> Data => data;

        public override TType this[params int[] indices]
        {
            get => TOp.ApplyCpu(Left[indices], Right[indices]);
            set => throw new System.NotImplementedException($"Cannot set value to {GetType().Name}");
        }
    }

}