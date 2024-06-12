namespace SharpGrad.Tensors.Operators
{
    public interface  IAggregator<TOperand1, TResult>: IExecutor
    {
        abstract static Shape ResultingShape(Shape operand1);
        abstract static TResult Exec(TOperand1[] operand1);
        abstract static TOperand1[] Backward(TOperand1[] operand1, TResult grad);
    }

}