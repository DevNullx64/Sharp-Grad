namespace SharpGrad.DifEngine.SyntaxBuilder.Operations
{
    /// <summary>
    /// Defines kinds of function nodes.
    /// </summary>
    public enum KindFunction
    {
        DifferentiableFunction = 3 | KindProperty.Differentiable,
        Function = 4,
    }
}
