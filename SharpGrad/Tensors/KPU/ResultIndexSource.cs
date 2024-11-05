namespace SharpGrad.Tensors.KPU
{
    public enum ResultIndexSource : short
    {
        //None = OperandIndexSource.None,
        //Operand = OperandIndexSource.Operand,
        //BroadcastOperand = OperandIndexSource.BroadcastOperand,
        //OperationIndeces = OperandIndexSource.OperationIndeces,
        Cache = OperandIndexSource.Cache,
        Output = OperandIndexSource.Output,

        MaxValue = OperandIndexSource.MaxValue
    }
}