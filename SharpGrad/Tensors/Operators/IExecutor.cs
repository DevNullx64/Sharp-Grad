namespace SharpGrad.Tensors.Operators
{
    public interface IExecutor
    {
        abstract static string Symbol { get; }
        /// <summary>
        /// The operation code.
        /// </summary>
        abstract static OpCode OpCode { get; }
    }

}