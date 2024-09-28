namespace SharpGrad.Tensors.Operators
{
    public interface  IAggregator<TOperand1, TResult>: IExec
    {
        abstract static Shape ResultingShape(Shape right);
        abstract static TResult Exec(TOperand1[] right);
        abstract static TOperand1[] Backward(TOperand1[] right, TResult grad);
    }

}