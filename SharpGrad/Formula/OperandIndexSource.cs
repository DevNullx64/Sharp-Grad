namespace SharpGrad.Formula
{
    public enum OperandIndexSource : short
    {
        None = -1,
        Operand = 0,
        BroadcastOperand = 1,
        Operation = 2,
        Cache = 3,
        Output = 4,

        MaxValue = Output + 1
    }
}