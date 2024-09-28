namespace SharpGrad.Tensors
{
    public enum KPUIndexSource : short
        {
            None = -1,
            Operand = 0,
            Operation = 1,
            Cache = 2,
            Output = 3
        }    

}