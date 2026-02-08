namespace SharpGrad.DifEngine.SyntaxBuilder
{
    /// <summary>
    /// Defines kinds of value nodes.
    /// </summary>
    public enum KindValue
    {
        Constant = 1 | KindCategory.Value | KindProperty.Differentiable,
        Variable = 2 | KindCategory.Value | KindProperty.Differentiable
    }
}
