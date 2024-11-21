using System.Numerics;

namespace SharpGrad.Tensors.Formula
{
    public class OutputData<TResult> where TResult : unmanaged, INumber<TResult>
    {
        private readonly ComputeElement<TResult> BaseElement;
        public TResult[] Value => BaseElement.IsComputed
            ? BaseElement.Result.Content!.SafeCPUData
            : BaseElement.Compute();

        public OutputData(ComputeElement<TResult> baseElement)
        {
            BaseElement = baseElement;
            BaseElement.IsOuput = true;
        }

        public static implicit operator ComputeElement<TResult>(OutputData<TResult> output)
            => output.BaseElement;

        public static implicit operator OutputData<TResult>(ComputeElement<TResult> element)
            => new(element);
    }
}