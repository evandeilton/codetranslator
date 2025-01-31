#include <iostream>
#include <vector>
#include <memory>

template <typename T>
class BinaryTree {
private:
    struct Node {
        T data;
        std::shared_ptr<Node> left;
        std::shared_ptr<Node> right;
        
        Node(T value) : data(value), left(nullptr), right(nullptr) {}
    };
    
    std::shared_ptr<Node> root;
    
    void insert_recursive(std::shared_ptr<Node>& node, T value) {
        if (!node) {
            node = std::make_shared<Node>(value);
            return;
        }
        
        if (value < node->data) {
            insert_recursive(node->left, value);
        } else {
            insert_recursive(node->right, value);
        }
    }

public:
    BinaryTree() : root(nullptr) {}
    
    void insert(T value) {
        insert_recursive(root, value);
    }
    
    bool contains(T value) {
        auto current = root;
        while (current) {
            if (current->data == value) return true;
            current = (value < current->data) ? current->left : current->right;
        }
        return false;
    }
};

int main() {
    BinaryTree<int> tree;
    std::vector<int> values = {5, 3, 7, 1, 9, 4, 6};
    
    for (int val : values) {
        tree.insert(val);
    }
    
    std::cout << "Contains 4? " << (tree.contains(4) ? "Yes" : "No") << std::endl;
    std::cout << "Contains 8? " << (tree.contains(8) ? "Yes" : "No") << std::endl;
    
    return 0;
}