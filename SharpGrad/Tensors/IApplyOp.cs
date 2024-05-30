namespace SharpGrad.Tensors
{
    public interface IApplyOp<TFrom, TTo> {
        abstract static Shape ResultShape(Shape left);
    }
}