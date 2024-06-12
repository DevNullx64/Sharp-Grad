namespace SharpGrad.Tensors.Operators
{
    public interface IExecutor
    {
        /// <summary>
        /// The operation code.
        /// </summary>
        abstract static OpCode OpCode { get; }
    }

}