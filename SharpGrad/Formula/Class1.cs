using SharpGrad.Operators;
using System.Numerics;

namespace SharpGrad.Formula
{
    public abstract class CUBase<T>
        where T : unmanaged, INumber<T>
    {
        public abstract T Value { get; }
    }

    public class Operand<T>(T value) : CUBase<T>
    where T : unmanaged, INumber<T>
    {
        public override T Value => value;

        public static Instruction<AddOp<T>, Operand<T>, Operand<T>, T>
            operator +(Operand<T> left, Operand<T> right)
            => new(left, right);

        public static Instruction<SubOp<T>, Operand<T>, Operand<T>, T>
            operator -(Operand<T> left, Operand<T> right)
            => new(left, right);

        public static Instruction<MulOp<T>, Operand<T>, Operand<T>, T>
            operator *(Operand<T> left, Operand<T> right)
            => new(left, right);

        public static Instruction<DivOp<T>, Operand<T>, Operand<T>, T>
            operator /(Operand<T> left, Operand<T> right)
            => new(left, right);
    }

    public abstract class Instruction<T> : CUBase<T>
        where T : unmanaged, INumber<T>
    { }

    public class Instruction<TOp, TLeft, T>(TLeft left) : Instruction<T>
        where TOp : IExecUnary<T, T>
        where TLeft : CUBase<T>
    where T : unmanaged, INumber<T>
    {
        public readonly TLeft Left = left;

        public override T Value => TOp.Invoke(Left.Value);

        public static Instruction<AddOp<T>, Instruction<TOp, TLeft, T>, Operand<T>, T>
            operator +(Instruction<TOp, TLeft, T> left, Operand<T> right)
            => new(left, right);
        public static Instruction<AddOp<T>, Operand<T>, Instruction<TOp, TLeft, T>, T>
            operator +(Operand<T> left, Instruction<TOp, TLeft, T> right)
            => new(left, right);

        public static Instruction<SubOp<T>, Instruction<TOp, TLeft, T>, Operand<T>, T>
            operator -(Instruction<TOp, TLeft, T> left, Operand<T> right)
            => new(left, right);
        public static Instruction<SubOp<T>, Operand<T>, Instruction<TOp, TLeft, T>, T>
            operator -(Operand<T> left, Instruction<TOp, TLeft, T> right)
            => new(left, right);

        public static Instruction<MulOp<T>, Instruction<TOp, TLeft, T>, Operand<T>, T>
            operator *(Instruction<TOp, TLeft, T> left, Operand<T> right)
            => new(left, right);
        public static Instruction<MulOp<T>, Operand<T>, Instruction<TOp, TLeft, T>, T>
            operator *(Operand<T> left, Instruction<TOp, TLeft, T> right)
            => new(left, right);

        public static Instruction<DivOp<T>, Instruction<TOp, TLeft, T>, Operand<T>, T>
            operator /(Instruction<TOp, TLeft, T> left, Operand<T> right)
            => new(left, right);
        public static Instruction<DivOp<T>, Operand<T>, Instruction<TOp, TLeft, T>, T>
            operator /(Operand<T> left, Instruction<TOp, TLeft, T> right)
            => new(left, right);
    }

    public class Instruction<TOp, TLeft, TRight, T>(TLeft left, TRight right) : Instruction<T>
        where TOp : IExecBinary<T, T, T>
        where TLeft : CUBase<T>
        where TRight : CUBase<T>
        where T : unmanaged, INumber<T>
    {
        public readonly TLeft Left = left;
        public readonly TRight Right = right;

        public override T Value => TOp.Invoke(Left.Value, Right.Value);

        public static Instruction<AddOp<T>, Instruction<TOp, TLeft, TRight, T>, Operand<T>, T>
            operator +(Instruction<TOp, TLeft, TRight, T> left, Operand<T> right)
            => new(left, right);
        public static Instruction<AddOp<T>, Operand<T>, Instruction<TOp, TLeft, TRight, T>, T>
            operator +(Operand<T> left, Instruction<TOp, TLeft, TRight, T> right)
            => new(left, right);

        public static Instruction<SubOp<T>, Instruction<TOp, TLeft, TRight, T>, Operand<T>, T>
            operator -(Instruction<TOp, TLeft, TRight, T> left, Operand<T> right)
            => new(left, right);
        public static Instruction<SubOp<T>, Operand<T>, Instruction<TOp, TLeft, TRight, T>, T>
            operator -(Operand<T> left, Instruction<TOp, TLeft, TRight, T> right)
            => new(left, right);

        public static Instruction<MulOp<T>, Instruction<TOp, TLeft, TRight, T>, Operand<T>, T>
            operator *(Instruction<TOp, TLeft, TRight, T> left, Operand<T> right)
            => new(left, right);
        public static Instruction<MulOp<T>, Operand<T>, Instruction<TOp, TLeft, TRight, T>, T>
            operator *(Operand<T> left, Instruction<TOp, TLeft, TRight, T> right)
            => new(left, right);

        public static Instruction<DivOp<T>, Instruction<TOp, TLeft, TRight, T>, Operand<T>, T> 
            operator /(Instruction<TOp, TLeft, TRight, T> left, Operand<T> right)
            => new(left, right);
        public static Instruction<DivOp<T>, Operand<T>, Instruction<TOp, TLeft, TRight, T>, T>
            operator /(Operand<T> left, Instruction<TOp, TLeft, TRight, T> right)
            => new(left, right);
    }

    // TODO: Review, it's not working as expected
    public class Instruction2<TOp, TLeft, TRight, T>(TLeft left, TRight right, T dumbValueClass) : Instruction<T>
        where TOp : IExecBinary<T, T, T>
        where TLeft : Instruction<T>
        where TRight : Instruction<T>
        where T : unmanaged, INumber<T>
    {
        public readonly TLeft Left = left;
        public readonly TRight Right = right;

        public override T Value => TOp.Invoke(Left.Value, Right.Value);

        public static Instruction<AddOp<T>, Instruction2<TOp, TLeft, TRight, T>, TLeft, T>
            operator +(Instruction2<TOp, TLeft, TRight, T> left, TLeft right)
            => new(left, right);
        public static Instruction<AddOp<T>, TLeft, Instruction2<TOp, TLeft, TRight, T>, T>
            operator +(TLeft left, Instruction2<TOp, TLeft, TRight, T> right)
            => new(left, right);
        public static Instruction<AddOp<T>, Instruction<TOp, TLeft, TRight, T>, Instruction2<TOp, TLeft, TRight, T>, T>
            operator +(Instruction<TOp, TLeft, TRight, T> left, Instruction2<TOp, TLeft, TRight, T> right)
            => new(left, right);
        public static Instruction<AddOp<T>, Instruction2<TOp, TLeft, TRight, T>, Instruction<TOp, TLeft, TRight, T>, T>
            operator +(Instruction2<TOp, TLeft, TRight, T> left, Instruction<TOp, TLeft, TRight, T> right)
            => new(left, right);
    }
}
