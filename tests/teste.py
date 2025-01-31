def bubble_sort(arr):
    """
    Implement bubble sort algorithm
    Args:
        arr: List of numbers to sort
    Returns:
        Sorted list
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# Test the function
test_array = [64, 34, 25, 12, 22, 11, 90]
sorted_array = bubble_sort(test_array)
print(f"Sorted array: {sorted_array}")