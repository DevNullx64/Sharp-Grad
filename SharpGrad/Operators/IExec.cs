namespace SharpGrad.Operators
{
    public interface IExec
    {
        /// <summary>
        /// The symbol, or name, of the operation.
        /// </summary>
        abstract static string Symbol { get; }

        /// <summary>
        /// The operation code.
        /// </summary>
        abstract static OpCode OpCode { get; }
    }

}