using System;
using System.Collections.Generic;

namespace SharpGrad
{
    /// <summary>
    /// Pool for dimension names and sizes.
    /// </summary>
    internal static class DimensionsPool
    {
        /// <summary>
        /// Maximum number of dimensions supported.
        /// </summary>
        public const int MaxDimensions = byte.MaxValue - 1;

        /// <summary>
        /// Index indicating that a dimension was not found.
        /// </summary>
        public const byte NotFound = byte.MaxValue;

        /// <summary>
        /// Lists of sizes and names of dimensions.
        /// </summary>
        public static List<int> Sizes = [1];

        /// <summary>
        /// List of names of dimensions.
        /// </summary>
        public static List<string> Names = ["Scalar"];

        /// <summary>
        /// Gets the name of a dimension by its index.
        /// </summary>
        public static string GetName(byte index) => Names[index];

        /// <summary>
        /// Gets the index of a dimension by its name.
        /// </summary>
        public static byte IndexOf(string name) => (byte)Names.IndexOf(name);

        /// <summary>
        /// Gets the size of a dimension by its index.
        /// </summary>
        public static int GetSize(byte index) => Sizes[index];

        /// <summary>
        /// Gets the index of a dimension by its name, adding it if it does not exist.
        /// </summary>
        /// <param name="name">The name of the dimension.</param>
        /// <param name="size">The size of the dimension.</param>
        /// <returns>The index of the dimension.</returns>
        /// <exception cref="ArgumentException">Thrown if the size is less than 2, if the name is null or whitespace, if the dimension already exists with a different size, or if too many dimensions are defined.</exception>
        public static byte GetOrAddDimension(string name, int size)
        {
            if (size < 2)
            {
                throw new ArgumentException($"Size must be greater than 1. Got {size}.", nameof(size));
            }
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("Dimension name cannot be null or whitespace.", nameof(name));
            }
            byte index = (byte)Names.IndexOf(name);
            if (index != NotFound)
            {
                if (Sizes[index] != size)
                {
                    throw new ArgumentException($"Dimension '{name}' already exists with size {Sizes[index]}, cannot redefine with size {size}.");
                }
                return index;
            }
            else
            {
                if (Names.Count == MaxDimensions)
                {
                    throw new ArgumentException($"Too many dimensions defined. Maximum is {MaxDimensions}.");
                }
                Names.Add(name);
                Sizes.Add(size);
                return (byte)(Names.Count - 1);
            }
        }
    }
}