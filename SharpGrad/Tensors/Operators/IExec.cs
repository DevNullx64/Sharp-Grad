using SharpGrad.Tensors.KPU;

namespace SharpGrad.Tensors.Operators
{
    public interface IExec
    {
        abstract static string Symbol { get; }
        /// <summary>
        /// The operation code.
        /// </summary>
        abstract static OpCode OpCode { get; }
    }

}