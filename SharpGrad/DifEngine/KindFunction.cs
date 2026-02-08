namespace SharpGrad.DifEngine.SyntaxBuilder
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
