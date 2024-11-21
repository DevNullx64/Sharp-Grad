using System.Numerics;

namespace SharpGrad.DifEngine
{
    public class DivValue<TType> : Value<TType>
        where TType : unmanaged, INumber<TType>
    {
        public DivValue(Value<TType> left, Value<TType> right)
            : base(left.Data / right.Data, "/", left, right)
        {
        }

        // TODO: Is SafeAccelerator a good way to backpropagate division?
        protected override void Backward()
        {
            LeftChildren!.Grad += Grad / RightChildren!.Data;
            RightChildren.Grad += Grad * LeftChildren.Data / (RightChildren.Data * RightChildren.Data);
        }
    }
}